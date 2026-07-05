// SPDX-License-Identifier: Apache-2.0
//! Vulkan compute engine.
//!
//! Provides:
//!  - Buffer allocation (device-local + host-visible staging)
//!  - Descriptor pool + set management
//!  - Command pool / command buffer recording
//!  - Synchronous dispatch: upload → compute → download

use ash::vk;
use std::sync::Arc;

use crate::pipeline::{Pipeline, PipelineCache, MAX_BINDINGS};

// ─── Buffer ───────────────────────────────────────────────────────────────────

/// A Vulkan buffer with its backing memory.
pub struct Buffer {
    device: ash::Device,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    /// Non-null when the memory is permanently mapped (host-visible).
    pub mapped_ptr: Option<*mut u8>,
    pub mem_props: vk::MemoryPropertyFlags,
}

// Safety: mapped_ptr is only accessed by the thread that owns the Buffer,
// and is only valid while the Buffer is alive.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn alloc(
        device: &ash::Device,
        pd: vk::PhysicalDevice,
        instance: &ash::Instance,
        size: u64,
        usage: vk::BufferUsageFlags,
        required_flags: vk::MemoryPropertyFlags,
        preferred_flags: vk::MemoryPropertyFlags,
    ) -> Result<Self, String> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_ci, None) }
            .map_err(|e| format!("create_buffer: {e}"))?;

        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };

        let memory_type = find_memory_type(&mem_props, req.memory_type_bits, required_flags, preferred_flags)
            .ok_or_else(|| {
                format!("No suitable memory type (req_flags={required_flags:?} pref_flags={preferred_flags:?})")
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type.index);
        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| format!("allocate_memory: {e}"))?;
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }
            .map_err(|e| format!("bind_buffer_memory: {e}"))?;

        // Permanently map if host-visible.
        let mapped_ptr = if memory_type.props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            let ptr = unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .map_err(|e| format!("map_memory: {e}"))? as *mut u8;
            Some(ptr)
        } else {
            None
        };

        Ok(Buffer {
            device: device.clone(),
            buffer,
            memory,
            size,
            mapped_ptr,
            mem_props: memory_type.props,
        })
    }

    /// Create a device-local buffer (for GPU-side tensors).
    pub fn device_local(
        device: &ash::Device,
        pd: vk::PhysicalDevice,
        instance: &ash::Instance,
        size: u64,
    ) -> Result<Self, String> {
        Self::alloc(
            device,
            pd,
            instance,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )
    }

    /// Create a host-visible staging buffer (for CPU ↔ GPU transfer).
    pub fn staging(
        device: &ash::Device,
        pd: vk::PhysicalDevice,
        instance: &ash::Instance,
        size: u64,
    ) -> Result<Self, String> {
        Self::alloc(
            device,
            pd,
            instance,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_CACHED,
        )
    }

    /// Write `data` into this buffer (only valid for host-visible buffers).
    pub fn write(&self, data: &[u8]) -> Result<(), String> {
        let ptr = self.mapped_ptr.ok_or("Buffer is not host-visible")?;
        if data.len() as u64 > self.size {
            return Err(format!(
                "Data len {} > buffer size {}",
                data.len(),
                self.size
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    /// Read `len` bytes from this buffer into `dst`.
    pub fn read(&self, dst: &mut [u8]) -> Result<(), String> {
        let ptr = self.mapped_ptr.ok_or("Buffer is not host-visible")?;
        if dst.len() as u64 > self.size {
            return Err(format!(
                "Dst len {} > buffer size {}",
                dst.len(),
                self.size
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped_ptr.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

// ─── Memory type helper ───────────────────────────────────────────────────────

struct MemType {
    index: u32,
    props: vk::MemoryPropertyFlags,
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
    preferred: vk::MemoryPropertyFlags,
) -> Option<MemType> {
    // Try required + preferred first, fall back to required only.
    for pass in [preferred | required, required] {
        for i in 0..mem_props.memory_type_count {
            if (type_bits & (1 << i)) == 0 {
                continue;
            }
            let props = mem_props.memory_types[i as usize].property_flags;
            if props.contains(pass) {
                return Some(MemType { index: i, props });
            }
        }
    }
    None
}

// ─── ComputeEngine ───────────────────────────────────────────────────────────

/// Descriptor pool chunk: 256 sets, 256 × MAX_BINDINGS storage buffers.
const POOL_SETS: u32 = 256;

/// Maximum number of pooled host-coherent buffers per size bucket.
const POOL_MAX: usize = 16;

/// A simple pool of reusable host-coherent storage buffers keyed by capacity.
/// Avoids per-activation malloc/mmap pressure during inference.
struct BufferPool {
    /// Maps buffer size → list of idle buffers.
    buckets: std::collections::HashMap<u64, Vec<Buffer>>,
}

impl BufferPool {
    fn new() -> Self {
        BufferPool { buckets: std::collections::HashMap::new() }
    }

    /// Return a buffer of at least `size` bytes, reusing one from the pool if
    /// available, otherwise allocating a fresh one.
    fn get(
        &mut self,
        device: &ash::Device,
        pd: vk::PhysicalDevice,
        instance: &ash::Instance,
        size: u64,
    ) -> Result<Buffer, String> {
        let bucket = self.buckets.entry(size).or_default();
        if let Some(buf) = bucket.pop() {
            return Ok(buf);
        }
        Buffer::alloc(
            device, pd, instance, size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_CACHED,
        )
    }

    /// Return a buffer to the pool for reuse. Discards it if the pool is full.
    fn put(&mut self, buf: Buffer) {
        let bucket = self.buckets.entry(buf.size).or_default();
        if bucket.len() < POOL_MAX {
            bucket.push(buf);
        }
        // If pool is full, buf is dropped here (freeing the Vulkan memory).
    }
}

/// A complete Vulkan compute environment: pipelines, pools, command recording.
pub struct ComputeEngine {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    compute_queue: vk::Queue,
    compute_queue_family: u32,
    pipeline_cache: PipelineCache,

    command_pool: vk::CommandPool,
    /// Single persistent command buffer, reused for every dispatch/batch via
    /// reset + begin instead of allocating (and freeing) a fresh command
    /// buffer from the pool on every call. vkAllocateCommandBuffers /
    /// vkFreeCommandBuffers involve driver bookkeeping on every call — on a
    /// translation layer such as KosmicKrisp (Vulkan-on-Metal) that cost is
    /// paid per decode-step dispatch, so reusing one command buffer removes
    /// it from the hot path entirely. Only one command buffer is ever in
    /// flight at a time (dispatches are fully synchronous, see
    /// `end_and_submit`), so a single persistent buffer is always safe to
    /// reset here.
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,

    descriptor_pools: Vec<vk::DescriptorPool>,
    /// Pre-allocated descriptor sets, consumed linearly.
    descriptor_sets: Vec<vk::DescriptorSet>,
    ds_cursor: usize,

    /// Pool of reusable host-coherent buffers for activation tensors.
    buf_pool: BufferPool,
}

impl ComputeEngine {
    pub fn new(
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        compute_queue: vk::Queue,
        compute_queue_family: u32,
        shaders: &std::collections::HashMap<&str, &[u8]>,
    ) -> Result<Self, String> {
        let pipeline_cache = PipelineCache::new(device.clone(), shaders)?;

        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_queue_family)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = unsafe { device.create_command_pool(&cmd_pool_ci, None) }
            .map_err(|e| format!("create_command_pool: {e}"))?;

        let cmd_buf_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = unsafe { device.allocate_command_buffers(&cmd_buf_alloc) }
            .map_err(|e| format!("allocate_command_buffers: {e}"))?[0];

        let fence_ci = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_ci, None) }
            .map_err(|e| format!("create_fence: {e}"))?;

        let mut engine = ComputeEngine {
            instance,
            physical_device,
            device,
            compute_queue,
            compute_queue_family,
            pipeline_cache,
            command_pool,
            cmd_buf,
            fence,
            descriptor_pools: Vec::new(),
            descriptor_sets: Vec::new(),
            ds_cursor: 0,
            buf_pool: BufferPool::new(),
        };

        // Pre-allocate an initial pool of descriptor sets.
        engine.grow_descriptor_pool()?;

        Ok(engine)
    }

    fn grow_descriptor_pool(&mut self) -> Result<(), String> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: POOL_SETS * MAX_BINDINGS,
        }];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(POOL_SETS);
        let pool = unsafe { self.device.create_descriptor_pool(&pool_ci, None) }
            .map_err(|e| format!("create_descriptor_pool: {e}"))?;

        let dsl = self.pipeline_cache.descriptor_set_layout;
        let dsls: Vec<vk::DescriptorSetLayout> = vec![dsl; POOL_SETS as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&dsls);
        let sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| format!("allocate_descriptor_sets: {e}"))?;

        self.descriptor_pools.push(pool);
        self.descriptor_sets.extend(sets);
        Ok(())
    }

    fn next_descriptor_set(&mut self) -> Result<vk::DescriptorSet, String> {
        if self.ds_cursor >= self.descriptor_sets.len() {
            self.grow_descriptor_pool()?;
        }
        let ds = self.descriptor_sets[self.ds_cursor];
        self.ds_cursor += 1;
        Ok(ds)
    }

    /// Reset descriptor set cursor (call at the start of each graph execution).
    pub fn reset_descriptor_sets(&mut self) {
        self.ds_cursor = 0;
    }

    /// Allocate a device-local buffer.
    pub fn alloc_device(&self, size: u64) -> Result<Buffer, String> {
        Buffer::device_local(
            &self.device,
            self.physical_device,
            &self.instance,
            size,
        )
    }

    /// Allocate a host-visible staging buffer.
    pub fn alloc_staging(&self, size: u64) -> Result<Buffer, String> {
        Buffer::staging(
            &self.device,
            self.physical_device,
            &self.instance,
            size,
        )
    }

    /// Allocate a host-coherent buffer usable as a storage buffer, drawn from
    /// the internal buffer pool to minimise repeated mmap/munmap calls.
    pub fn alloc_host_coherent_storage(&mut self, size: u64) -> Result<Buffer, String> {
        self.buf_pool.get(&self.device, self.physical_device, &self.instance, size)
    }

    /// Return a host-coherent buffer to the pool for reuse.
    pub fn return_to_pool(&mut self, buf: Buffer) {
        self.buf_pool.put(buf);
    }

    /// Upload `data` to a device-local buffer via a staging buffer.
    pub fn upload(&mut self, dst: &Buffer, data: &[u8]) -> Result<(), String> {
        let staging = self.alloc_staging(data.len() as u64)?;
        staging.write(data)?;

        let cb = self.begin_one_shot()?;
        let copy = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(data.len() as u64);
        unsafe {
            self.device
                .cmd_copy_buffer(cb, staging.buffer, dst.buffer, &[copy]);
        }
        self.end_and_submit(cb)?;

        // staging dropped here, freeing the host buffer.
        Ok(())
    }

    /// Download `len` bytes from `src` into `out`.
    pub fn download(&mut self, src: &Buffer, out: &mut Vec<u8>) -> Result<(), String> {
        let size = src.size;
        let staging = self.alloc_staging(size)?;

        let cb = self.begin_one_shot()?;
        let copy = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(size);
        unsafe {
            self.device
                .cmd_copy_buffer(cb, src.buffer, staging.buffer, &[copy]);
        }
        self.end_and_submit(cb)?;

        out.resize(size as usize, 0);
        staging.read(out)?;
        Ok(())
    }

    /// Reset and begin the single persistent command buffer for a new
    /// recording. Avoids a vkAllocateCommandBuffers call on every dispatch —
    /// see the `cmd_buf` field doc comment for why this matters here.
    fn begin_one_shot(&self) -> Result<vk::CommandBuffer, String> {
        unsafe {
            self.device
                .reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())
        }
        .map_err(|e| format!("reset_command_buffer: {e}"))?;

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(self.cmd_buf, &begin) }
            .map_err(|e| format!("begin_command_buffer: {e}"))?;

        Ok(self.cmd_buf)
    }

    fn end_and_submit(&self, cb: vk::CommandBuffer) -> Result<(), String> {
        unsafe { self.device.end_command_buffer(cb) }
            .map_err(|e| format!("end_command_buffer: {e}"))?;

        let submit = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cb));
        unsafe {
            self.device.queue_submit(self.compute_queue, &[submit], self.fence)
        }
        .map_err(|e| format!("queue_submit: {e}"))?;

        unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
        }
        .map_err(|e| format!("wait_for_fences: {e}"))?;

        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|e| format!("reset_fences: {e}"))?;

        // Command buffer is NOT freed — it's a persistent handle owned by
        // `self.cmd_buf`, reset and reused on the next `begin_one_shot`.

        Ok(())
    }

    // ─── Batched dispatch primitives ─────────────────────────────────────

    /// Open a command buffer for batched recording.
    ///
    /// Use `record_to` to add dispatches and `record_barrier_to` between
    /// dependent ops, then `submit_batch` to submit everything at once.
    pub fn begin_batch(&self) -> Result<vk::CommandBuffer, String> {
        self.begin_one_shot()
    }

    /// Record a compute dispatch into an open command buffer WITHOUT submitting.
    ///
    /// Must be called between `begin_batch` and `submit_batch`.
    pub fn record_to(
        &mut self,
        cb: vk::CommandBuffer,
        shader_name: &str,
        buffers: &[&Buffer],
        push_constants: &[u8],
        workgroups: (u32, u32, u32),
    ) -> Result<(), String> {
        let (vk_pipeline, vk_layout) = {
            let pipeline = self
                .pipeline_cache
                .get(shader_name)
                .ok_or_else(|| format!("Shader '{shader_name}' not found"))?;
            (pipeline.pipeline, pipeline.layout)
        };

        let ds = self.next_descriptor_set()?;

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| vk::DescriptorBufferInfo::default()
                .buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE))
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos.iter().enumerate()
            .map(|(i, info)| vk::WriteDescriptorSet::default()
                .dst_set(ds).dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info)))
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        unsafe {
            if !push_constants.is_empty() {
                self.device.cmd_push_constants(cb, vk_layout, vk::ShaderStageFlags::COMPUTE, 0, push_constants);
            }
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, vk_pipeline);
            self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, vk_layout, 0, &[ds], &[]);
            self.device.cmd_dispatch(cb, workgroups.0, workgroups.1, workgroups.2);
        }
        Ok(())
    }

    /// Insert a compute-to-compute memory barrier into an open command buffer.
    ///
    /// Required between two dispatches when the second reads data written by
    /// the first (e.g. RMSNorm output fed into a MatVec input).
    pub fn record_barrier_to(&self, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );
        }
    }

    /// Submit and wait for a command buffer built with `begin_batch` + `record_to`.
    ///
    /// After the fence signals, descriptor sets are safe to reuse — reset the
    /// cursor so subsequent batches reuse the same pre-allocated pool entries
    /// instead of growing the pool indefinitely.
    pub fn submit_batch(&mut self, cb: vk::CommandBuffer) -> Result<(), String> {
        self.end_and_submit(cb)?;
        // GPU is fully idle now (fence waited). Reuse descriptor sets from the top.
        self.ds_cursor = 0;
        Ok(())
    }

    // ─── Single-op dispatch (wraps begin_batch + record_to + submit_batch) ─

    /// Execute a single compute dispatch synchronously.
    ///
    /// Parameters:
    /// - `shader_name`: name of the SPIR-V variant (e.g. `"silu_f32"`)
    /// - `buffers`: storage buffers bound to bindings 0..N
    /// - `push_constants`: raw bytes written to push-constant range
    /// - `workgroups`: (x, y, z) workgroup dispatch counts
    pub fn dispatch(
        &mut self,
        shader_name: &str,
        buffers: &[&Buffer],
        push_constants: &[u8],
        workgroups: (u32, u32, u32),
    ) -> Result<(), String> {
        let cb = self.begin_one_shot()?;
        self.record_to(cb, shader_name, buffers, push_constants, workgroups)?;
        self.end_and_submit(cb)?;
        self.ds_cursor = 0;
        Ok(())
    }

    /// List compiled shader names.
    pub fn available_shaders(&self) -> Vec<String> {
        self.pipeline_cache.pipeline_names()
    }
}

impl Drop for ComputeEngine {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for &pool in &self.descriptor_pools {
                self.device.destroy_descriptor_pool(pool, None);
            }
            self.device.destroy_fence(self.fence, None);
            // destroy_command_pool implicitly frees self.cmd_buf too.
            self.device.destroy_command_pool(self.command_pool, None);
            // pipeline_cache and device are dropped after this.
        }
    }
}
