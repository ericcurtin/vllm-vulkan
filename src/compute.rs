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
    /// Logical size presented to callers — the number of bytes the current
    /// holder requested. All reads/writes/readbacks bound against this.
    pub size: u64,
    /// Physical VkBuffer allocation size, `>= size`. Equal to `size` for direct
    /// allocations; larger only for pooled buffers whose request was rounded up
    /// to a coarse size class (see `size_class`) so the pool buckets by a bounded
    /// set of capacities instead of leaking one bucket per distinct length.
    pub capacity: u64,
    /// Non-null when the memory is permanently mapped (host-visible).
    pub mapped_ptr: Option<*mut u8>,
    pub mem_props: vk::MemoryPropertyFlags,
}

// Safety invariant: a Buffer's mapped_ptr is only read/written on the single
// engine-owning thread (the GIL-holding forward thread). rayon parallelism in
// the model layer operates exclusively on host Vec<f32> slices and never
// captures a Buffer, so no &Buffer is shared for concurrent mutation. Send+Sync
// are asserted on that basis; write()/read() are &self to allow the
// raw-pointer activation-buffer scheme. Debug builds assert the owning thread
// in ComputeEngine (see creator_thread).
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
            capacity: size,
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

    /// Write `data` into this host-visible buffer at byte `offset` (for filling a
    /// large resident buffer slice-by-slice without a full host-side concat — the
    /// step3p7 GPU expert loader streams each expert straight into GTT this way).
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), String> {
        let ptr = self.mapped_ptr.ok_or("Buffer is not host-visible")?;
        if offset + data.len() as u64 > self.size {
            return Err(format!(
                "write_at: offset {} + len {} > buffer size {}",
                offset, data.len(), self.size
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());
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
///
/// Sized for the union of two demands, whichever is larger per size bucket:
///
/// 1. The fork's WS2 fused MoE submit holds 36 `moe_inter`-sized buffers at
///    once (8 experts × {gate,up,act,mid} + the shared expert's 4). The
///    previous cap of 16 made every MoE layer-call vkAllocate ~20 fresh
///    buffers (recorded in moe_fused_submit) and free them again on return
///    (moe_readback) — ~45 ms per 12 steps of pure allocator churn on the
///    BC-250.
///
/// 2. Upstream's vLLM continuous-batching decode-attention path:
///    `_paged_attn_decode_batch` (kv_ops.py) builds one `execute_batch` op —
///    and hence one temp + one output buffer of the same size — per
///    concurrently-decoding sequence, all in a single call, once per
///    attention layer per decode step. Concurrent decode batch sizes of
///    32-256 sequences are ordinary under real serving load; at cap 16 every
///    buffer past the 16th forced a fresh `Buffer::alloc` (~2.2-3.4us) vs a
///    ~210ns pool hit — a ~10-16x per-buffer regression.
///
/// A bucket only ever grows to the max simultaneously-live count for its
/// size, so the higher cap costs nothing for sizes that never fan out this
/// widely. 256 covers both the MoE fan-out and realistic decode batches; the
/// bounded per-bucket memory cost (256 × buffer_size, all host-coherent/
/// UMA-resident, not discrete VRAM) is modest.
const POOL_MAX: usize = 256;

/// Round an allocation request UP to a coarse size class so the number of
/// distinct pool buckets stays bounded (O(log n)) even when a caller requests a
/// steadily-growing size on every step.
///
/// This is load-bearing for the DSV4 long-context decode: the attention path
/// re-projects its FULL O(t) token history through the resident matvec seam
/// every token (`attention_layer_decode` → `mm(compressor.*, x_hist, t1, …)`),
/// so `gpu_matvec_rows` requests a `t1·H·4`-byte scratch whose size GROWS by one
/// row per token and is then `return_to_pool`'d. Keyed by the exact byte size,
/// the pool accreted a fresh, never-freed host-coherent (GTT-resident) buffer
/// for every distinct history length — an O(t²) leak of wired GTT memory that
/// exhausts the ~13 GB GTT and surfaces as `amdgpu: Not enough memory for
/// command submission` / `amdgpu_vm_validate() failed`. Classing lets successive
/// lengths reuse one bucket; the byte budget below bounds what stays resident.
///
/// Classing is transparent to the math: a pooled buffer's `capacity` is the
/// classed (possibly larger) allocation, but its `size` is reset to the exact
/// logical request on every `get`, so all reads/writes/readbacks see the logical
/// length — results are bit-for-bit identical (argmax-exact).
fn size_class(size: u64) -> u64 {
    // Small buffers (the many seq=1 decode activations): fine 256 B granularity,
    // negligible waste. Larger buffers: next power of two — caps the live bucket
    // count at ~log2(max) while never wasting more than 2× on a transient that is
    // then reused across steps instead of leaked.
    const SMALL: u64 = 64 * 1024; // 64 KiB
    if size <= SMALL {
        (size + 255) & !255
    } else {
        size.next_power_of_two()
    }
}

/// Total idle bytes the pool may retain across all buckets before returned
/// buffers are FREED instead of cached. Bounds the GTT held by idle scratch so a
/// long decode's growing-history projections cannot accumulate unbounded wired
/// memory. Simultaneously-live buffers are checked OUT of the pool and are not
/// counted here, so a wide fan-out (MoE experts / concurrent decode batch) is
/// unaffected — only the idle free-list is capped. Override with
/// `VLLM_VULKAN_POOL_BUDGET_MB` (default 1 GiB).
fn pool_budget_bytes() -> u64 {
    static B: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("VLLM_VULKAN_POOL_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(1024 * 1024 * 1024)
    })
}

/// A simple pool of reusable host-coherent storage buffers keyed by capacity.
/// Avoids per-activation malloc/mmap pressure during inference. Requests are
/// size-classed (`size_class`) so a monotonically growing request length reuses
/// a bounded set of buckets, and the retained idle set is capped by
/// `pool_budget_bytes()` so it cannot grow without bound over a long decode.
struct BufferPool {
    /// Maps CLASSED capacity → list of idle buffers of that capacity.
    buckets: std::collections::HashMap<u64, Vec<Buffer>>,
    /// Sum of the capacities of all idle buffers currently held (bookkeeping for
    /// the byte budget).
    idle_bytes: u64,
}

impl BufferPool {
    fn new() -> Self {
        BufferPool { buckets: std::collections::HashMap::new(), idle_bytes: 0 }
    }

    /// Return a buffer of at least `size` bytes, reusing one from the pool if
    /// available, otherwise allocating a fresh one at the classed capacity. The
    /// returned buffer's `size` is set to the exact logical request; its
    /// `capacity` may be larger (the class).
    fn get(
        &mut self,
        device: &ash::Device,
        pd: vk::PhysicalDevice,
        instance: &ash::Instance,
        size: u64,
    ) -> Result<Buffer, String> {
        let cap = size_class(size);
        let bucket = self.buckets.entry(cap).or_default();
        if let Some(mut buf) = bucket.pop() {
            // buf.capacity == cap; present the caller's exact logical size.
            self.idle_bytes = self.idle_bytes.saturating_sub(cap);
            buf.size = size;
            return Ok(buf);
        }
        let mut buf = Buffer::alloc(
            device, pd, instance, cap,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_CACHED,
        )?;
        // Allocated at the class capacity; expose the exact logical size.
        buf.size = size;
        Ok(buf)
    }

    /// Return a buffer to the pool for reuse, keyed by its (classed) capacity.
    /// Discards it (freeing the Vulkan memory) if the per-bucket cap OR the
    /// global idle-byte budget would be exceeded.
    fn put(&mut self, buf: Buffer) {
        let cap = buf.capacity;
        if self.idle_bytes.saturating_add(cap) <= pool_budget_bytes() {
            let bucket = self.buckets.entry(cap).or_default();
            if bucket.len() < POOL_MAX {
                bucket.push(buf);
                self.idle_bytes += cap;
                return;
            }
        }
        // Over budget or bucket full: buf is dropped here, freeing the memory.
    }
}

/// Command-buffer/fence ring depth for the pipelined decode path.
const CB_RING_DEPTH: usize = 2;

struct RingSlot {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    in_flight: bool,
    /// Pool buffers to return only once this slot's fence signals (empty for the
    /// resident decode path, which uses persistent buffers).
    deferred_returns: Vec<Buffer>,
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
    /// Persistent command buffer for the batch path (begin_batch/submit_batch).
    /// Submits are serial (the fence blocks), so one CB is reset+reused instead
    /// of allocate+free per submit — that alloc/free was a measurable chunk of
    /// the ~88µs/submit decode overhead (decode is submit-bound, not bandwidth).
    batch_cb: vk::CommandBuffer,

    descriptor_pools: Vec<vk::DescriptorPool>,
    /// Pre-allocated descriptor sets, consumed linearly.
    descriptor_sets: Vec<vk::DescriptorSet>,
    ds_cursor: usize,

    /// Pool of reusable host-coherent buffers for activation tensors.
    buf_pool: BufferPool,

    /// True between begin_batch and submit_batch. One-shot ops (upload/download/
    /// dispatch) share the fence + descriptor cursor and would corrupt an open
    /// batch recording, so they refuse to run while this is set.
    batch_open: std::sync::atomic::AtomicBool,

    /// The thread that constructed this engine. Buffer::mapped_ptr is only
    /// sound to touch from this thread (see the Buffer Send/Sync safety
    /// comment) — debug builds assert it at every buffer-touching entry point.
    creator_thread: std::thread::ThreadId,

    ring: Vec<RingSlot>,          // len CB_RING_DEPTH; unused when ring disabled
    ring_enabled: bool,
    ring_cursor: usize,           // next slot index (monotonic; %DEPTH picks slot)
    ring_cur_slot: usize,         // slot of the open pipelined batch
    ring_last_submitted: Option<usize>,

    /// GPU timestamp query pool (VLLM_VULKAN_Q35_TSTAMP attribution): lazily
    /// created; None until first `ensure_ts_pool`. `ts_period` is the device's
    /// nanoseconds-per-tick (`limits.timestamp_period`); 0.0 ⇒ timestamps
    /// unsupported on this queue family (pool stays unusable).
    ts_pool: Option<vk::QueryPool>,
    ts_capacity: u32,
    ts_period: f64,
}

impl ComputeEngine {
    pub fn new(
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        compute_queue: vk::Queue,
        compute_queue_family: u32,
        caps: crate::device::DeviceCaps,
        shaders: &std::collections::HashMap<&str, &[u8]>,
    ) -> Result<Self, String> {
        let pipeline_cache = PipelineCache::new(device.clone(), caps, shaders)?;

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

        // One persistent command buffer reused across all batch submits.
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let batch_cb = unsafe { device.allocate_command_buffers(&cb_alloc) }
            .map_err(|e| format!("allocate batch_cb: {e}"))?[0];

        let ring_enabled = crate::flags::flags_global().cb_ring;
        let mut ring = Vec::with_capacity(CB_RING_DEPTH);
        if ring_enabled {
            let cb_alloc_ring = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(CB_RING_DEPTH as u32);
            let cbs = unsafe { device.allocate_command_buffers(&cb_alloc_ring) }
                .map_err(|e| format!("allocate ring CBs: {e}"))?;
            for &cb in &cbs {
                let f = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                    .map_err(|e| format!("create ring fence: {e}"))?;
                ring.push(RingSlot { cb, fence: f, in_flight: false, deferred_returns: Vec::new() });
            }
        }

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
            batch_cb,
            descriptor_pools: Vec::new(),
            descriptor_sets: Vec::new(),
            ds_cursor: 0,
            buf_pool: BufferPool::new(),
            batch_open: std::sync::atomic::AtomicBool::new(false),
            creator_thread: std::thread::current().id(),
            ring,
            ring_enabled,
            ring_cursor: 0,
            ring_cur_slot: 0,
            ring_last_submitted: None,
            ts_pool: None,
            ts_capacity: 0,
            ts_period: 0.0,
        };

        // Pre-allocate an initial pool of descriptor sets.
        engine.grow_descriptor_pool()?;

        Ok(engine)
    }

    /// Debug-only check that we're running on the engine's owning thread.
    /// See the Buffer Send/Sync safety comment: mapped_ptr access off this
    /// thread is unsound. Cheap no-op in release builds.
    fn assert_owner(&self) {
        debug_assert_eq!(
            std::thread::current().id(),
            self.creator_thread,
            "ComputeEngine used off its owning thread — Buffer mapped_ptr is not thread-safe"
        );
    }

    /// Compile an additional pipeline variant with custom specialization
    /// constants, beyond what's registered by name at construction time.
    /// pub(crate)-only: used by exploratory tests to measure whether a
    /// given BLOCK_SIZE/NUM_ROWS/NUM_COLS combination is worth adding as
    /// a real, permanent variant before committing to it (see
    /// `pipeline::PipelineCache::compile_one_with_spec`).
    #[cfg(test)]
    pub(crate) fn compile_extra_variant(
        &mut self, name: &str, spv: &[u8], spec_constants: &[(u32, u32)],
    ) -> Result<(), String> {
        self.pipeline_cache.compile_one_with_spec(name, spv, spec_constants)
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
        self.assert_owner();
        if self.batch_open.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("one-shot op called while a batch is open (begin_batch without submit_batch)".to_string());
        }
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
        self.assert_owner();
        if self.batch_open.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("one-shot op called while a batch is open (begin_batch without submit_batch)".to_string());
        }
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

        self.wait_fence_gil(self.fence)?;

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
        self.assert_owner();
        // Reset + reuse the persistent CB (no allocate/free per submit).
        unsafe {
            self.device
                .reset_command_buffer(self.batch_cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset batch_cb: {e}"))?;
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(self.batch_cb, &begin) }
            .map_err(|e| format!("begin batch_cb: {e}"))?;
        self.batch_open.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(self.batch_cb)
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
        self.assert_owner();
        let (vk_pipeline, vk_layout) = {
            let pipeline = self
                .pipeline_cache
                .get(shader_name)
                .ok_or_else(|| format!("Shader '{shader_name}' not found"))?;
            (pipeline.pipeline, pipeline.layout)
        };

        let ds = self.next_descriptor_set()?;

        // `buffers.len()` is always <= MAX_BINDINGS (the descriptor set
        // layout has exactly that many bindings — see pipeline.rs), so
        // both of these can be fixed-size stack arrays instead of heap-
        // allocated `Vec`s that this function rebuilt from scratch on
        // every single dispatch (several hundred times per decode step in
        // the decode hot path). `buffer_infos` must outlive the
        // `update_descriptor_sets` call below since each `writes[i]`
        // stores a raw pointer into it (via `.buffer_info(...)`) — same
        // requirement the original `Vec`-based version had, just now
        // satisfied by a stack array that isn't moved instead of a heap
        // allocation that wasn't either.
        let n = buffers.len();
        // A real (not debug-only) check: `debug_assert!` is compiled out in
        // release builds, which would leave the out-of-bounds
        // `buffer_infos[i]`/`writes[i]` array writes below as the only
        // thing standing between an over-large `buffers` slice and a
        // panic — Rust's array indexing always bounds-checks (so this
        // could never become silent memory corruption), but it's a worse,
        // less diagnosable failure mode than returning the `Err` this
        // function's signature already supports for its other error
        // paths (e.g. "Shader not found" above).
        if n > MAX_BINDINGS as usize {
            return Err(format!(
                "record_to: {n} buffers exceeds MAX_BINDINGS ({MAX_BINDINGS})"
            ));
        }

        let mut buffer_infos = [vk::DescriptorBufferInfo::default(); MAX_BINDINGS as usize];
        for (i, b) in buffers.iter().enumerate() {
            buffer_infos[i] = vk::DescriptorBufferInfo::default()
                .buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE);
        }

        let mut writes = [vk::WriteDescriptorSet::default(); MAX_BINDINGS as usize];
        for i in 0..n {
            writes[i] = vk::WriteDescriptorSet::default()
                .dst_set(ds).dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[i]));
        }
        unsafe { self.device.update_descriptor_sets(&writes[..n], &[]) };

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

    /// Like `record_to`, but each binding carries its own BYTE offset into the
    /// buffer instead of the hardcoded `offset(0)`. Needed for per-token slices
    /// of a T-wide activation buffer (batched-verify RoPE/attention: token i's
    /// Q/ATTN slice at `i*row_bytes`, POS's i-th entry at `i*4`) where `record_to`
    /// would always bind the buffer's start. `record_to` remains the offset-0
    /// special case for every other (whole-buffer / row-batched) dispatch.
    pub fn record_to_off(
        &mut self,
        cb: vk::CommandBuffer,
        shader_name: &str,
        buffers: &[(&Buffer, u64)],
        push_constants: &[u8],
        workgroups: (u32, u32, u32),
    ) -> Result<(), String> {
        self.assert_owner();
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
            .map(|(b, off)| vk::DescriptorBufferInfo::default()
                .buffer(b.buffer).offset(*off).range(vk::WHOLE_SIZE))
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

    /// Record a `vkCmdCopyBuffer` (one region) into an open command buffer
    /// WITHOUT submitting. Used to copy a freshly-computed K/V slice from a
    /// compute output buffer into the resident KV cache plane, in the SAME
    /// command buffer as the surrounding dispatches (1-CB unified layer).
    /// `src`/`dst` must both carry TRANSFER_SRC/TRANSFER_DST usage (the pool
    /// buffers do). Offsets/size are in BYTES.
    pub fn record_copy_to(
        &self,
        cb: vk::CommandBuffer,
        src: &Buffer,
        dst: &Buffer,
        src_off: u64,
        dst_off: u64,
        size: u64,
    ) {
        let copy = vk::BufferCopy::default()
            .src_offset(src_off)
            .dst_offset(dst_off)
            .size(size);
        unsafe {
            self.device.cmd_copy_buffer(cb, src.buffer, dst.buffer, &[copy]);
        }
    }

    /// Insert a TRANSFER→COMPUTE barrier into an open command buffer.
    ///
    /// Required when a recorded `vkCmdCopyBuffer` (TRANSFER stage, TRANSFER_WRITE)
    /// writes data that a subsequent dispatch reads (COMPUTE stage, SHADER_READ),
    /// e.g. the resident-KV copy feeding the paged-attention decode dispatch.
    /// (`record_barrier_to` only covers COMPUTE→COMPUTE.)
    pub fn record_transfer_to_compute_barrier(&self, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );
        }
    }

    /// Insert a COMPUTE→TRANSFER barrier into an open command buffer.
    ///
    /// Required when a dispatch (COMPUTE stage, SHADER_WRITE) produces data that
    /// a subsequent `vkCmdCopyBuffer` (TRANSFER stage, TRANSFER_READ) reads,
    /// e.g. RoPE writing UR_K/UR_V before the resident-KV copy reads them.
    pub fn record_compute_to_transfer_barrier(&self, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );
        }
    }

    /// Block until `fence` signals. Releases the Python GIL for the duration when
    /// enabled AND a Python interpreter is live (so pure-Rust tests, which have no
    /// interpreter, take the direct path). Honours VLLM_VULKAN_SPIN.
    fn wait_fence_gil(&self, fence: vk::Fence) -> Result<(), String> {
        let device = &self.device;
        let spin = crate::flags::flags_global().spin;
        let wait = move || -> Result<(), String> {
            if spin {
                loop {
                    match unsafe { device.get_fence_status(fence) } {
                        Ok(true) => break,
                        Ok(false) => std::hint::spin_loop(),
                        Err(e) => return Err(format!("get_fence_status: {e}")),
                    }
                }
                Ok(())
            } else {
                unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }
                    .map_err(|e| format!("wait_for_fences: {e}"))
            }
        };
        let want_release = crate::flags::flags_global().gil_release
            && unsafe { pyo3::ffi::Py_IsInitialized() != 0 };
        if want_release {
            // We hold the GIL (called from a pymethod stack); with_gil hands back a
            // token cheaply, allow_threads releases it around the blocking wait.
            // Closure captures only Ungil data (&ash::Device, Copy handles).
            pyo3::Python::with_gil(|py| py.allow_threads(wait))
        } else {
            wait()
        }
    }

    /// Submit and wait for a command buffer built with `begin_batch` + `record_to`.
    ///
    /// After the fence signals, descriptor sets are safe to reuse — reset the
    /// cursor so subsequent batches reuse the same pre-allocated pool entries
    /// instead of growing the pool indefinitely.
    pub fn submit_batch(&mut self, cb: vk::CommandBuffer) -> Result<(), String> {
        self.assert_owner();
        // PROFILE split (submit-tax study): eng_qsubmit = CPU IB-build+queue ioctl
        // (the RADV per-submit tax direct-submit attacks); eng_fence = blocking
        // GPU-exec drain (NOT direct-submit-addressable).
        let _t_qs = std::time::Instant::now();
        // End + submit + wait, but DO NOT free the CB (reused next begin_batch).
        unsafe { self.device.end_command_buffer(cb) }
            .map_err(|e| format!("end batch_cb: {e}"))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
        unsafe { self.device.queue_submit(self.compute_queue, &[submit], self.fence) }
            .map_err(|e| format!("queue_submit: {e}"))?;
        crate::prof_add("eng_qsubmit", _t_qs);
        // Blocking fence wait. (Busy-polling get_fence_status was measured to
        // give ~0% over this — the per-submit cost is the GPU dispatch
        // launch/drain, not the fence wake-up latency — so it's not worth the
        // CPU burn. Opt into a spin with VLLM_VULKAN_SPIN=1 for experiments.)
        let _t_fw = std::time::Instant::now();
        self.wait_fence_gil(self.fence)?;
        crate::prof_add("eng_fence", _t_fw);
        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|e| format!("reset_fences: {e}"))?;
        // GPU is fully idle now (fence waited). Reuse descriptor sets from the top.
        self.ds_cursor = 0;
        self.batch_open.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// NON-BLOCKING submit: end + queue_submit the batch CB, then RETURN
    /// immediately WITHOUT waiting the fence. The caller MUST call `wait_batch`
    /// before touching any buffer the CB writes or recording a new batch — the
    /// descriptor sets and output buffers are in use by the still-in-flight GPU
    /// work until then. `batch_open` and `ds_cursor` are intentionally left as-is
    /// (the descriptors are live) and get reset in `wait_batch`.
    ///
    /// Purpose (Laguna CPU-overlap lever): submit the routed-expert command
    /// buffer, run the data-independent shared expert on the host rayon pool
    /// concurrently with the GPU, then `wait_batch` + read back. The overlap
    /// changes only WHEN the host shared branch computes, never the reduction
    /// order, so it stays bit-exact with the sequential path.
    pub fn submit_batch_async(&mut self, cb: vk::CommandBuffer) -> Result<(), String> {
        self.assert_owner();
        let _t_qs = std::time::Instant::now();
        unsafe { self.device.end_command_buffer(cb) }
            .map_err(|e| format!("end batch_cb (async): {e}"))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
        unsafe { self.device.queue_submit(self.compute_queue, &[submit], self.fence) }
            .map_err(|e| format!("queue_submit (async): {e}"))?;
        crate::prof_add("eng_qsubmit", _t_qs);
        Ok(())
    }

    /// Block until the in-flight `submit_batch_async` fence signals, then reset
    /// the fence + descriptor cursor (mirrors the tail of `submit_batch`). After
    /// this returns the GPU is idle and the batch's output buffers are safe to
    /// read; the next `begin_batch` reuses the descriptor pool from the top.
    pub fn wait_batch(&mut self) -> Result<(), String> {
        self.assert_owner();
        let _t_fw = std::time::Instant::now();
        self.wait_fence_gil(self.fence)?;
        crate::prof_add("eng_fence", _t_fw);
        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|e| format!("reset_fences (async): {e}"))?;
        self.ds_cursor = 0;
        self.batch_open.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    // ─── GPU timestamp queries (VLLM_VULKAN_Q35_TSTAMP attribution) ────────

    /// Lazily create a timestamp query pool of `n` slots. Returns true when
    /// timestamps are usable (pool exists AND the compute queue family exposes
    /// valid timestamp bits AND `timestamp_period > 0`). Best-effort: any
    /// failure leaves timestamps disabled rather than erroring the forward.
    pub fn ensure_ts_pool(&mut self, n: u32) -> bool {
        if self.ts_pool.is_some() {
            return self.ts_capacity >= n && self.ts_period > 0.0;
        }
        let qfp = unsafe {
            self.instance.get_physical_device_queue_family_properties(self.physical_device)
        };
        let valid_bits = qfp.get(self.compute_queue_family as usize)
            .map(|p| p.timestamp_valid_bits).unwrap_or(0);
        let period = unsafe {
            self.instance.get_physical_device_properties(self.physical_device)
        }.limits.timestamp_period as f64;
        if valid_bits == 0 || period <= 0.0 {
            log::warn!("timestamp queries unsupported on compute queue family (valid_bits={valid_bits}, period={period})");
            return false;
        }
        let ci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(n);
        match unsafe { self.device.create_query_pool(&ci, None) } {
            Ok(pool) => {
                self.ts_pool = Some(pool);
                self.ts_capacity = n;
                self.ts_period = period;
                true
            }
            Err(e) => {
                log::warn!("create_query_pool(TIMESTAMP,{n}): {e}");
                false
            }
        }
    }

    /// Reset query slots [first, first+count) inside an open CB. Must precede
    /// any `ts_cmd_mark` writes to those slots in the same CB.
    pub fn ts_cmd_reset(&self, cb: vk::CommandBuffer, first: u32, count: u32) {
        if let Some(pool) = self.ts_pool {
            unsafe { self.device.cmd_reset_query_pool(cb, pool, first, count) };
        }
    }

    /// Write a timestamp into slot `idx`. `top=true` uses TOP_OF_PIPE (a start
    /// marker before any work); otherwise BOTTOM_OF_PIPE (completes when all
    /// previously recorded commands have fully drained — with the full compute
    /// barriers between phases this brackets each phase's GPU execution).
    pub fn ts_cmd_mark(&self, cb: vk::CommandBuffer, idx: u32, top: bool) {
        if let Some(pool) = self.ts_pool {
            let stage = if top { vk::PipelineStageFlags::TOP_OF_PIPE }
                        else { vk::PipelineStageFlags::BOTTOM_OF_PIPE };
            unsafe { self.device.cmd_write_timestamp(cb, stage, pool, idx) };
        }
    }

    /// Read back `count` timestamps from `first` as NANOSECONDS (tick ×
    /// timestamp_period). Call only after the CB's fence has been waited.
    pub fn ts_read_ns(&self, first: u32, count: u32) -> Result<Vec<f64>, String> {
        let pool = self.ts_pool.ok_or("ts_read_ns: no query pool")?;
        let mut ticks = vec![0u64; count as usize];
        unsafe {
            self.device.get_query_pool_results(
                pool, first, &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }.map_err(|e| format!("get_query_pool_results: {e}"))?;
        Ok(ticks.iter().map(|&t| t as f64 * self.ts_period).collect())
    }

    // ─── Pipelined CB ring (M5b) — resident decode path only ───────────────

    /// Is the CB ring active? (flag on AND slots allocated.)
    pub fn ring_active(&self) -> bool {
        self.ring_enabled && !self.ring.is_empty()
    }

    /// Start a token on the ring: drain any leftover in-flight slots and reset the
    /// descriptor cursor exactly once. Call at the top of a resident forward.
    pub fn begin_forward_ring(&mut self) -> Result<(), String> {
        self.assert_owner();
        for slot in 0..self.ring.len() {
            if self.ring[slot].in_flight { self.wait_ring_slot(slot)?; }
        }
        self.ds_cursor = 0;
        self.ring_cursor = 0;
        self.ring_last_submitted = None;
        Ok(())
    }

    /// Open a ring command buffer for pipelined recording. Waits the target slot's
    /// fence only if the CPU has lapped the ring (slot still in flight from
    /// CB_RING_DEPTH submits ago). Does NOT reset ds_cursor.
    pub fn begin_batch_pipelined(&mut self) -> Result<vk::CommandBuffer, String> {
        self.assert_owner();
        debug_assert!(self.ring_active(), "begin_batch_pipelined without an active ring");
        let slot = self.ring_cursor % self.ring.len();
        if self.ring[slot].in_flight { self.wait_ring_slot(slot)?; }
        let cb = self.ring[slot].cb;
        unsafe {
            self.device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset ring cb: {e}"))?;
        }
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cb, &begin) }
            .map_err(|e| format!("begin ring cb: {e}"))?;
        self.ring_cur_slot = slot;
        self.batch_open.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(cb)
    }

    /// Submit a pipelined CB WITHOUT waiting. `deferred` buffers are returned to the
    /// pool only after this slot's fence signals (pass an empty Vec for the
    /// persistent-buffer decode path).
    pub fn submit_batch_pipelined(
        &mut self,
        cb: vk::CommandBuffer,
        deferred: Vec<Buffer>,
    ) -> Result<(), String> {
        self.assert_owner();
        let slot = self.ring_cur_slot;
        unsafe { self.device.end_command_buffer(cb) }
            .map_err(|e| format!("end ring cb: {e}"))?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
        unsafe { self.device.queue_submit(self.compute_queue, &[submit], self.ring[slot].fence) }
            .map_err(|e| format!("queue_submit ring: {e}"))?;
        self.ring[slot].in_flight = true;
        self.ring[slot].deferred_returns = deferred;
        self.ring_last_submitted = Some(slot);
        self.ring_cursor += 1;
        self.batch_open.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Wait until the most-recently-submitted pipelined CB completes. Call before a
    /// host read of that CB's output (q/k/v after CB1, logits after the final CB).
    pub fn wait_batch_pipelined(&mut self) -> Result<(), String> {
        self.assert_owner();
        if let Some(slot) = self.ring_last_submitted {
            if self.ring[slot].in_flight { self.wait_ring_slot(slot)?; }
        }
        Ok(())
    }

    /// Wait a slot's fence (GIL-released), reset it, clear in_flight, drain deferred
    /// pool returns.
    fn wait_ring_slot(&mut self, slot: usize) -> Result<(), String> {
        let fence = self.ring[slot].fence;
        self.wait_fence_gil(fence)?;                       // reuses the M5a helper
        unsafe { self.device.reset_fences(&[fence]) }
            .map_err(|e| format!("reset ring fence: {e}"))?;
        self.ring[slot].in_flight = false;
        let returns = std::mem::take(&mut self.ring[slot].deferred_returns);
        for buf in returns { self.buf_pool.put(buf); }
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
        self.assert_owner();
        if self.batch_open.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("one-shot op called while a batch is open (begin_batch without submit_batch)".to_string());
        }
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

    /// Whether a compiled pipeline named `name` exists in this engine's cache.
    pub fn has_pipeline(&self, name: &str) -> bool {
        self.pipeline_cache.get(name).is_some()
    }

    /// Runtime compile of ONE extra shader variant under the G2 creation
    /// watchdog (see `PipelineCache::compile_with_spec_timeout`). Used by the
    /// `debug_*_geometry` sweep tools to probe new BLOCK_SIZE geometries
    /// without hanging the process on a GFX1013 driver-compile hang.
    /// Ok(true)=ready, Ok(false)=creation timed out (variant skipped), Err=fail.
    pub fn compile_variant_timeout(
        &mut self,
        name: &str,
        spv: &[u8],
        spec: &[(u32, u32)],
        timeout_ms: u64,
    ) -> Result<bool, String> {
        self.pipeline_cache.compile_with_spec_timeout(
            name, spv, spec, std::time::Duration::from_millis(timeout_ms),
        )
    }
}

impl Drop for ComputeEngine {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for slot in &self.ring {
                self.device.destroy_fence(slot.fence, None);
            }
            for &pool in &self.descriptor_pools {
                self.device.destroy_descriptor_pool(pool, None);
            }
            if let Some(pool) = self.ts_pool {
                self.device.destroy_query_pool(pool, None);
            }
            self.device.destroy_fence(self.fence, None);
            // destroy_command_pool implicitly frees self.cmd_buf too.
            self.device.destroy_command_pool(self.command_pool, None);
            // pipeline_cache and device are dropped after this.
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::{size_class, POOL_MAX};

    #[test]
    fn size_class_is_a_valid_capacity() {
        // Never smaller than the request (would corrupt writes), and idempotent
        // (a classed size classes to itself, so put()/get() round-trip stably).
        for &s in &[1u64, 4, 255, 256, 257, 4096, 65_535, 65_536, 65_537, 1 << 20, 3_000_000] {
            let c = size_class(s);
            assert!(c >= s, "class {c} < request {s}");
            assert_eq!(size_class(c), c, "class not idempotent at {s}");
        }
    }

    #[test]
    fn growing_lengths_collapse_to_a_bounded_bucket_set() {
        // The DSV4 leak shape: a request length that grows by one row every
        // decode step. Exact-size keying minted one bucket per step (O(t)); the
        // size classes must collapse these to O(log) distinct capacities so the
        // idle pool stays bounded. H*4 bytes per row, 4096 hidden.
        let row = 4096u64 * 4;
        let classes: std::collections::HashSet<u64> =
            (1..=2048u64).map(|t| size_class(t * row)).collect();
        // 2048 distinct exact sizes must collapse to a small (log-scale) set.
        assert!(classes.len() <= 24, "buckets not bounded: {}", classes.len());
    }

    #[test]
    fn small_requests_waste_little() {
        // Sub-64KiB decode activations round to 256 B — at most 255 B waste.
        for s in [1u64, 100, 4096, 40_000, 65_536] {
            assert!(size_class(s) - s < 256, "excess waste at {s}");
        }
        // Guard the per-bucket cap constant stays sane alongside the byte budget.
        assert!(POOL_MAX >= 1);
    }
}
