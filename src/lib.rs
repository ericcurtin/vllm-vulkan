// SPDX-License-Identifier: Apache-2.0
//! vLLM-Vulkan Rust extension.
//!
//! On macOS, Vulkan calls are translated to Metal by KosmicKrisp
//! (Mesa/Zink software Vulkan driver).  On Linux x86_64 and aarch64,
//! native Vulkan is used directly.
//!
//! Provides GPU-accelerated LLM tensor operations via Vulkan compute shaders
//! copied from the llama.cpp project (MIT-licensed GLSL, compiled to SPIR-V
//! at build time).
//!
//! The PyO3 module `_rs` is the bridge between this crate and the Python
//! package `vllm_vulkan`.

mod device;
mod pipeline;
mod compute;
pub mod model;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub use device::VulkanDevice;

// ─── Device query functions ──────────────────────────────────────────────────

/// Return True if at least one Vulkan-capable device is present.
#[pyfunction]
fn is_available() -> bool {
    device::is_vulkan_available()
}

/// Number of Vulkan-capable physical devices.
#[pyfunction]
fn get_device_count() -> usize {
    device::device_count()
}

/// Enumerate all Vulkan physical devices as a list of dicts.
#[pyfunction]
fn enumerate_devices(py: Python<'_>) -> PyResult<Vec<PyObject>> {
    device::enumerate_devices()
        .into_iter()
        .map(|info| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("name", &info.name)?;
            dict.set_item("vendor_id", info.vendor_id)?;
            dict.set_item("device_type", &info.device_type)?;
            dict.set_item("api_version", &info.api_version)?;
            dict.set_item("driver_version", info.driver_version)?;
            dict.set_item("total_memory_bytes", info.total_memory_bytes)?;
            Ok(dict.into())
        })
        .collect()
}

/// Return device info dict for `device_idx`.
#[pyfunction]
fn get_device_info(py: Python<'_>, device_idx: usize) -> PyResult<PyObject> {
    let devs = device::enumerate_devices();
    let info = devs
        .get(device_idx)
        .ok_or_else(|| PyRuntimeError::new_err(format!("no device at index {device_idx}")))?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("name", &info.name)?;
    dict.set_item("vendor_id", info.vendor_id)?;
    dict.set_item("device_type", &info.device_type)?;
    dict.set_item("api_version", &info.api_version)?;
    dict.set_item("driver_version", info.driver_version)?;
    dict.set_item("total_memory_bytes", info.total_memory_bytes)?;
    Ok(dict.into())
}

/// Synchronise all pending Vulkan work across all devices.
#[pyfunction]
fn synchronize() -> PyResult<()> {
    device::synchronize_all().map_err(PyRuntimeError::new_err)
}

/// Return `(used_bytes, total_bytes)` for `device_idx`.
#[pyfunction]
fn get_memory_info(device_idx: usize) -> PyResult<(u64, u64)> {
    device::memory_info(device_idx).map_err(PyRuntimeError::new_err)
}

/// Temperature/top-p/top-k sample a token id from a full vocab of logits.
///
/// Exposed standalone (not just via `VulkanModel.forward_and_sample`) so
/// existing call sites that already have a logits list/array in hand (or
/// tests) can use the fast Rust sampler directly instead of Python's
/// `sorted()`-based `temperature_sample` — see `model::sample_with_temperature`
/// for why that matters. `uniform_random` should be a fresh uniform `[0, 1)`
/// draw per call (e.g. Python's `random.random()`); `top_k <= 0` means no
/// top-k filtering (use the full vocab); `temperature < 0.01` means greedy
/// (argmax) sampling, ignoring `top_p`/`top_k`/`uniform_random` (see
/// `model::sample_with_temperature`'s doc comment for why 0.01 specifically).
#[pyfunction]
#[pyo3(signature = (logits, temperature=1.0, top_p=1.0, top_k=64, uniform_random=0.0))]
fn sample_logits(logits: Vec<f32>, temperature: f32, top_p: f32, top_k: i64, uniform_random: f32) -> PyResult<usize> {
    // Unlike VulkanModel.forward_and_sample (whose logits always come from
    // this crate's own forward_gpu/forward, which never return an empty
    // vector), this function accepts an arbitrary caller-supplied `logits`
    // directly from Python — an empty list would otherwise reach
    // model::sample_with_temperature's final `nucleus.last().unwrap()` and
    // panic (aborting the whole Python process, not raising a catchable
    // exception) rather than failing gracefully.
    if logits.is_empty() {
        return Err(PyRuntimeError::new_err("sample_logits: logits must not be empty"));
    }
    Ok(model::sample_with_temperature(&logits, temperature, top_p, top_k, uniform_random))
}

// ─── GpuTensor — persistent device-local buffer ──────────────────────────────

/// A tensor resident on the Vulkan device.
///
/// Weights are uploaded once at model-load time and reused across every
/// forward pass, eliminating per-call upload overhead.
#[pyclass]
pub struct GpuTensor {
    buf: compute::Buffer,
    pub nbytes: u64,
}

#[pymethods]
impl GpuTensor {
    /// Number of bytes stored in this buffer.
    #[getter]
    fn nbytes(&self) -> u64 {
        self.nbytes
    }

    fn __repr__(&self) -> String {
        format!("GpuTensor({} bytes)", self.nbytes)
    }
}

// ─── VulkanContext — Python-accessible compute engine ────────────────────────

/// A live Vulkan compute context for a specific physical device.
///
/// Holds the logical device, pipeline cache, command pool, and descriptor
/// pools needed to dispatch GPU compute shaders.
#[pyclass]
pub struct VulkanContext {
    engine: compute::ComputeEngine,
}

#[pymethods]
impl VulkanContext {
    /// Create a VulkanContext for the device at `device_idx`.
    ///
    /// This compiles all pre-built SPIR-V shaders into Vulkan pipelines.
    #[new]
    #[pyo3(signature = (device_idx = 0))]
    fn new(device_idx: usize) -> PyResult<Self> {
        let dev = device::ComputeDevice::create(device_idx)
            .map_err(PyRuntimeError::new_err)?;

        // Load all pre-compiled SPIR-V shaders.
        let shader_spvs = include_all_shaders();

        let refs: std::collections::HashMap<&str, &[u8]> =
            shader_spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();

        let engine = compute::ComputeEngine::new(
            dev.instance.clone(),
            dev.physical_device,
            dev.device.clone(),
            dev.compute_queue,
            dev.compute_queue_family,
            &refs,
        )
        .map_err(PyRuntimeError::new_err)?;

        Ok(VulkanContext { engine })
    }

    /// Return the list of compiled shader names available on this context.
    fn available_shaders(&self) -> Vec<String> {
        self.engine.available_shaders()
    }

    /// Upload `data` bytes to a persistent device-local GPU buffer and return
    /// a `GpuTensor` handle.  The data is never downloaded back unless
    /// explicitly requested via `download_tensor`.
    fn upload_tensor(&mut self, data: &[u8]) -> PyResult<GpuTensor> {
        let buf = self.engine
            .alloc_device(data.len() as u64)
            .map_err(PyRuntimeError::new_err)?;
        self.engine
            .upload(&buf, data)
            .map_err(PyRuntimeError::new_err)?;
        Ok(GpuTensor { nbytes: buf.size, buf })
    }

    /// Execute a shader with a mix of persistent `GpuTensor` inputs and fresh
    /// byte-slice inputs, in a caller-specified binding order.
    ///
    /// `bindings` is a list of `(GpuTensor | bytes)` items in binding order.
    /// Pass a `GpuTensor` for persistent weight buffers; pass `bytes` for
    /// activations that change every call.  Output slots follow after all inputs.
    ///
    /// Returns output buffers as Python bytes.
    #[pyo3(signature = (shader_name, bindings, output_sizes, push_constants, workgroups))]
    fn execute_mixed(
        &mut self,
        py: Python<'_>,
        shader_name: &str,
        bindings: Vec<PyObject>,
        output_sizes: Vec<u64>,
        push_constants: Vec<u8>,
        workgroups: (u32, u32, u32),
    ) -> PyResult<Vec<PyObject>> {
        // Resolve each binding to a Buffer — either the persistent GpuTensor
        // buffer or a freshly-allocated-and-uploaded temporary.
        let mut temp_bufs: Vec<compute::Buffer> = Vec::new();
        let mut buf_ptrs: Vec<usize> = Vec::new(); // indices into either temp_bufs or gpu_tensors

        enum BufRef { Temp(usize), Gpu(*const compute::Buffer) }
        let mut refs: Vec<BufRef> = Vec::new();

        for obj in &bindings {
            Python::with_gil(|py_inner| -> PyResult<()> {
                if let Ok(gt) = obj.downcast_bound::<GpuTensor>(py_inner) {
                    let ptr = &gt.borrow().buf as *const compute::Buffer;
                    refs.push(BufRef::Gpu(ptr));
                 } else if let Ok(bytes) = obj.downcast_bound::<pyo3::types::PyBytes>(py_inner) {
                     let data = bytes.as_bytes();
                     // Use host-coherent storage for temporary activation buffers.
                     // On UMA/integrated GPUs (Grace-Blackwell) this avoids consuming
                     // device-local VRAM for every activation tensor.
                     let buf = self.engine
                         .alloc_host_coherent_storage(data.len() as u64)
                         .map_err(PyRuntimeError::new_err)?;
                     buf.write(data).map_err(PyRuntimeError::new_err)?;
                     let idx = temp_bufs.len();
                     temp_bufs.push(buf);
                     refs.push(BufRef::Temp(idx));
                } else {
                    return Err(PyRuntimeError::new_err(
                        "execute_mixed: each binding must be a GpuTensor or bytes"
                    ));
                }
                Ok(())
            })?;
        }

        // Allocate output buffers as host-coherent so we can read them directly
        // without a separate staging download step.
        let out_bufs: Vec<compute::Buffer> = output_sizes
            .iter()
            .map(|&sz| self.engine.alloc_host_coherent_storage(sz).map_err(PyRuntimeError::new_err))
            .collect::<PyResult<_>>()?;

        // Build the full binding list.
        let mut all_refs: Vec<&compute::Buffer> = Vec::new();
        for r in &refs {
            match r {
                BufRef::Temp(i) => all_refs.push(&temp_bufs[*i]),
                // Safety: the GpuTensor outlives this call (it's borrowed from Python)
                BufRef::Gpu(p) => all_refs.push(unsafe { &**p }),
            }
        }
        for o in &out_bufs {
            all_refs.push(o);
        }

        self.engine
            .dispatch(shader_name, &all_refs, &push_constants, workgroups)
            .map_err(PyRuntimeError::new_err)?;

        // Read outputs directly (host-coherent, no staging needed) then return
        // all temporary buffers to the pool for reuse.
        let result: PyResult<Vec<PyObject>> = out_bufs
            .iter()
            .map(|buf| {
                let mut data = vec![0u8; buf.size as usize];
                buf.read(&mut data).map_err(PyRuntimeError::new_err)?;
                Ok(pyo3::types::PyBytes::new(py, &data).into())
            })
            .collect();

        // Return temporary buffers to the pool AFTER reading outputs.
        for buf in temp_bufs {
            self.engine.return_to_pool(buf);
        }
        for buf in out_bufs {
            self.engine.return_to_pool(buf);
        }

        result
    }

    /// Execute multiple compute dispatches in a SINGLE `vkQueueSubmit`.
    ///
    /// This is the performance-critical path for transformer inference.
    /// Instead of N × (submit + wait) = N × ~150µs driver overhead, we pay
    /// that cost once for a whole batch of ops.
    ///
    /// Each element of `ops` is a tuple:
    ///   (shader_name, bindings, output_sizes, push_constants, workgroups, barrier)
    /// where:
    ///   - shader_name:   &str
    ///   - bindings:      list[GpuTensor | bytes]   — in binding-index order
    ///   - output_sizes:  list[int]                 — byte sizes of output slots
    ///   - push_constants: bytes
    ///   - workgroups:    (int, int, int)
    ///   - barrier:       bool  — insert compute→compute barrier AFTER this op
    ///
    /// Returns list (one per op) of lists (one per output) of bytes.
    #[pyo3(signature = (ops))]
    fn execute_batch(
        &mut self,
        py: Python<'_>,
        ops: Vec<(
            String,                     // shader_name
            Vec<PyObject>,              // bindings
            Vec<u64>,                   // output_sizes
            Vec<u8>,                    // push_constants
            (u32, u32, u32),            // workgroups
            bool,                       // barrier_after
        )>,
    ) -> PyResult<Vec<Vec<PyObject>>> {
        // ── Phase 1: allocate all buffers BEFORE recording ──────────────
        // VkBuffer handles must be stable for the duration of the command buffer.
        // We collect temp bufs (from byte inputs) and out bufs separately, then
        // build a flat mapping for each op.

        struct OpBuffers {
            // Indices into global temp_bufs / out_bufs vectors.
            temp_indices: Vec<usize>,   // one per bytes binding, in order
            out_start:    usize,        // first output buf index in out_bufs
            out_count:    usize,
            barrier_after: bool,
        }

        let mut temp_bufs: Vec<compute::Buffer> = Vec::new();
        let mut out_bufs: Vec<compute::Buffer> = Vec::new();
        let mut op_meta: Vec<OpBuffers> = Vec::new();

        for (_, bindings, output_sizes, _, _, barrier) in &ops {
            let mut temp_indices = Vec::new();
            for binding in bindings {
                Python::with_gil(|py_inner| -> PyResult<()> {
                    if binding.downcast_bound::<GpuTensor>(py_inner).is_ok() {
                        // GpuTensor — no temp buffer needed; handled during record.
                    } else if let Ok(bytes) = binding.downcast_bound::<pyo3::types::PyBytes>(py_inner) {
                        let data = bytes.as_bytes();
                        let buf = self.engine
                            .alloc_host_coherent_storage(data.len() as u64)
                            .map_err(PyRuntimeError::new_err)?;
                        buf.write(data).map_err(PyRuntimeError::new_err)?;
                        temp_indices.push(temp_bufs.len());
                        temp_bufs.push(buf);
                    } else {
                        return Err(PyRuntimeError::new_err(
                            "execute_batch: each binding must be GpuTensor or bytes"
                        ));
                    }
                    Ok(())
                })?;
            }
            let out_start = out_bufs.len();
            for &sz in output_sizes {
                out_bufs.push(
                    self.engine
                        .alloc_host_coherent_storage(sz)
                        .map_err(PyRuntimeError::new_err)?
                );
            }
            op_meta.push(OpBuffers {
                temp_indices,
                out_start,
                out_count: output_sizes.len(),
                barrier_after: *barrier,
            });
        }

        // ── Phase 2: record all dispatches into one command buffer ───────
        let cb = self.engine.begin_batch().map_err(PyRuntimeError::new_err)?;

        for ((shader_name, bindings, _, push_constants, workgroups, _), meta) in
            ops.iter().zip(op_meta.iter())
        {
            // Build the &Buffer slice in binding order.
            let mut all_refs: Vec<&compute::Buffer> = Vec::new();
            let mut temp_cursor = 0usize;
            for binding in bindings {
                Python::with_gil(|py_inner| -> PyResult<()> {
                    if let Ok(gt) = binding.downcast_bound::<GpuTensor>(py_inner) {
                        // Safety: GpuTensor Python object lives for the call duration.
                        let ptr = &gt.borrow().buf as *const compute::Buffer;
                        all_refs.push(unsafe { &*ptr });
                    } else {
                        all_refs.push(&temp_bufs[meta.temp_indices[temp_cursor]]);
                        temp_cursor += 1;
                    }
                    Ok(())
                })?;
            }
            // Output buffers come after inputs.
            for i in 0..meta.out_count {
                all_refs.push(&out_bufs[meta.out_start + i]);
            }

            self.engine
                .record_to(cb, shader_name, &all_refs, push_constants, *workgroups)
                .map_err(PyRuntimeError::new_err)?;

            if meta.barrier_after {
                self.engine.record_barrier_to(cb);
            }
        }

        // ── Phase 3: single submit+wait ───────────────────────────────────
        self.engine.submit_batch(cb).map_err(PyRuntimeError::new_err)?;

        // ── Phase 4: read outputs ─────────────────────────────────────────
        let mut all_results: Vec<Vec<PyObject>> = Vec::with_capacity(op_meta.len());
        for meta in &op_meta {
            let mut op_out: Vec<PyObject> = Vec::with_capacity(meta.out_count);
            for i in 0..meta.out_count {
                let buf = &out_bufs[meta.out_start + i];
                let mut data = vec![0u8; buf.size as usize];
                buf.read(&mut data).map_err(PyRuntimeError::new_err)?;
                op_out.push(pyo3::types::PyBytes::new(py, &data).into());
            }
            all_results.push(op_out);
        }

        // Return buffers to pool.
        for buf in temp_bufs { self.engine.return_to_pool(buf); }
        for buf in out_bufs   { self.engine.return_to_pool(buf); }

        Ok(all_results)
    }

    /// Allocate a persistent host-coherent buffer for activation tensors.
    ///
    /// On UMA systems (GB10) this is directly GPU-accessible with no DMA.
    /// Use `update_activation` to rewrite contents, pass as a binding in
    /// `execute_mixed` or `execute_batch`.
    fn alloc_activation(&mut self, nbytes: u64) -> PyResult<GpuTensor> {
        let buf = self.engine
            .alloc_host_coherent_storage(nbytes)
            .map_err(PyRuntimeError::new_err)?;
        Ok(GpuTensor { nbytes: buf.size, buf })
    }

    /// Overwrite a persistent activation buffer in-place (single memcpy).
    fn update_activation(&self, tensor: &mut GpuTensor, data: &[u8]) -> PyResult<()> {
        tensor.buf.write(data).map_err(PyRuntimeError::new_err)
    }

    /// Read back a persistent activation buffer.
    fn read_activation<'py>(&self, py: Python<'py>, tensor: &GpuTensor) -> PyResult<PyObject> {
        let mut data = vec![0u8; tensor.nbytes as usize];
        tensor.buf.read(&mut data).map_err(PyRuntimeError::new_err)?;
        Ok(pyo3::types::PyBytes::new(py, &data).into())
    }

    /// Execute two chained compute ops in one vkQueueSubmit where Op 1's input
    /// is Op 0's output (e.g. RMSNorm → MatVec in a transformer layer).
    ///
    /// Parameters:
    ///   shader0, bindings0, output_size0, pc0, wg0 — first op
    ///   shader1, bindings1, output_size1, pc1, wg1 — second op
    ///
    /// The output of shader0 is automatically used as an additional input
    /// binding (appended LAST) to shader1.  bindings1 should NOT include
    /// the intermediate buffer — it will be added automatically.
    ///
    /// Returns (output0_bytes, output1_bytes).
    #[pyo3(signature = (shader0, bindings0, output_size0, pc0, wg0,
                        shader1, bindings1, output_size1, pc1, wg1))]
    #[allow(clippy::too_many_arguments)]
    fn execute_chained(
        &mut self,
        py: Python<'_>,
        shader0: &str,
        bindings0: Vec<PyObject>,
        output_size0: u64,
        pc0: Vec<u8>,
        wg0: (u32, u32, u32),
        shader1: &str,
        bindings1: Vec<PyObject>,
        output_size1: u64,
        pc1: Vec<u8>,
        wg1: (u32, u32, u32),
    ) -> PyResult<(PyObject, PyObject)> {
        // ── Allocate all buffers ──────────────────────────────────────────
        let mut temp_bufs: Vec<compute::Buffer> = Vec::new();

        // Resolve bindings for Op 0.
        let mut refs0: Vec<*const compute::Buffer> = Vec::new();
        for binding in &bindings0 {
            Python::with_gil(|py_inner| -> PyResult<()> {
                if let Ok(gt) = binding.downcast_bound::<GpuTensor>(py_inner) {
                    refs0.push(&gt.borrow().buf as *const compute::Buffer);
                } else if let Ok(bytes) = binding.downcast_bound::<pyo3::types::PyBytes>(py_inner) {
                    let data = bytes.as_bytes();
                    let buf = self.engine.alloc_host_coherent_storage(data.len() as u64)
                        .map_err(PyRuntimeError::new_err)?;
                    buf.write(data).map_err(PyRuntimeError::new_err)?;
                    refs0.push(&buf as *const compute::Buffer);
                    temp_bufs.push(buf);
                } else {
                    return Err(PyRuntimeError::new_err("binding must be GpuTensor or bytes"));
                }
                Ok(())
            })?;
        }

        // Intermediate buffer: Op 0 writes here, Op 1 reads here.
        let inter_buf = self.engine.alloc_host_coherent_storage(output_size0)
            .map_err(PyRuntimeError::new_err)?;
        refs0.push(&inter_buf as *const compute::Buffer);

        // Op 0 output: same as inter_buf (we read it back for debugging if needed).
        // Op 1 output.
        let out1_buf = self.engine.alloc_host_coherent_storage(output_size1)
            .map_err(PyRuntimeError::new_err)?;

        // Resolve bindings for Op 1 (NOT including the intermediate — added below).
        let mut refs1: Vec<*const compute::Buffer> = Vec::new();
        for binding in &bindings1 {
            Python::with_gil(|py_inner| -> PyResult<()> {
                if let Ok(gt) = binding.downcast_bound::<GpuTensor>(py_inner) {
                    refs1.push(&gt.borrow().buf as *const compute::Buffer);
                } else if let Ok(bytes) = binding.downcast_bound::<pyo3::types::PyBytes>(py_inner) {
                    let data = bytes.as_bytes();
                    let buf = self.engine.alloc_host_coherent_storage(data.len() as u64)
                        .map_err(PyRuntimeError::new_err)?;
                    buf.write(data).map_err(PyRuntimeError::new_err)?;
                    refs1.push(&buf as *const compute::Buffer);
                    temp_bufs.push(buf);
                } else {
                    return Err(PyRuntimeError::new_err("binding must be GpuTensor or bytes"));
                }
                Ok(())
            })?;
        }
        // Append inter_buf as input to Op 1, then out1_buf as output.
        refs1.push(&inter_buf as *const compute::Buffer);
        refs1.push(&out1_buf as *const compute::Buffer);

        // ── Record both dispatches into one command buffer ────────────────
        let cb = self.engine.begin_batch().map_err(PyRuntimeError::new_err)?;

        // Op 0: resolve raw pointers to &Buffer.
        {
            let buf_refs0: Vec<&compute::Buffer> = refs0.iter()
                .map(|p| unsafe { &**p })
                .collect();
            self.engine.record_to(cb, shader0, &buf_refs0, &pc0, wg0)
                .map_err(PyRuntimeError::new_err)?;
            self.engine.record_barrier_to(cb);  // Op 1 reads Op 0's output
        }

        // Op 1.
        {
            let buf_refs1: Vec<&compute::Buffer> = refs1.iter()
                .map(|p| unsafe { &**p })
                .collect();
            self.engine.record_to(cb, shader1, &buf_refs1, &pc1, wg1)
                .map_err(PyRuntimeError::new_err)?;
        }

        // ── Single submit+wait ────────────────────────────────────────────
        self.engine.submit_batch(cb).map_err(PyRuntimeError::new_err)?;

        // ── Read outputs ──────────────────────────────────────────────────
        // Op 0 output (inter_buf) — caller may not need this, but return anyway.
        let mut data0 = vec![0u8; output_size0 as usize];
        inter_buf.read(&mut data0).map_err(PyRuntimeError::new_err)?;

        let mut data1 = vec![0u8; output_size1 as usize];
        out1_buf.read(&mut data1).map_err(PyRuntimeError::new_err)?;

        // Return all temporary and output buffers to pool.
        for buf in temp_bufs { self.engine.return_to_pool(buf); }
        self.engine.return_to_pool(inter_buf);
        self.engine.return_to_pool(out1_buf);

        Ok((
            pyo3::types::PyBytes::new(py, &data0).into(),
            pyo3::types::PyBytes::new(py, &data1).into(),
        ))
    }

    /// Execute a compute shader synchronously.
    ///
    /// Args:
    ///     shader_name: Name of the SPIR-V variant (e.g. `"silu_f32"`).
    ///     inputs: List of byte buffers — GPU inputs (uploaded before dispatch).
    ///     output_sizes: List of output buffer sizes in bytes.
    ///     push_constants: Raw bytes for the push-constant block (up to 128 bytes).
    ///     workgroups: (x, y, z) workgroup dispatch counts.
    ///
    /// Returns:
    ///     List of output buffers as Python bytes objects.
    #[pyo3(signature = (shader_name, inputs, output_sizes, push_constants, workgroups))]
    fn execute(
        &mut self,
        py: Python<'_>,
        shader_name: &str,
        inputs: Vec<Vec<u8>>,
        output_sizes: Vec<u64>,
        push_constants: Vec<u8>,
        workgroups: (u32, u32, u32),
    ) -> PyResult<Vec<PyObject>> {
        // Allocate input buffers on the GPU and upload data.
        let in_bufs: Vec<compute::Buffer> = inputs
            .iter()
            .map(|data| {
                let buf = self
                    .engine
                    .alloc_device(data.len() as u64)
                    .map_err(PyRuntimeError::new_err)?;
                self.engine
                    .upload(&buf, data)
                    .map_err(PyRuntimeError::new_err)?;
                Ok(buf)
            })
            .collect::<PyResult<_>>()?;

        // Allocate output buffers.
        let out_bufs: Vec<compute::Buffer> = output_sizes
            .iter()
            .map(|&sz| {
                self.engine
                    .alloc_device(sz)
                    .map_err(PyRuntimeError::new_err)
            })
            .collect::<PyResult<_>>()?;

        // Collect all buffer references.
        let all_refs: Vec<&compute::Buffer> = in_bufs
            .iter()
            .chain(out_bufs.iter())
            .collect();

        // Dispatch the shader.
        self.engine
            .dispatch(shader_name, &all_refs, &push_constants, workgroups)
            .map_err(PyRuntimeError::new_err)?;

        // Download outputs and return as Python bytes.
        out_bufs
            .iter()
            .map(|buf| {
                let mut data = Vec::new();
                self.engine
                    .download(buf, &mut data)
                    .map_err(PyRuntimeError::new_err)?;
                Ok(pyo3::types::PyBytes::new(py, &data).into())
            })
            .collect()
    }
}

// ─── SPIR-V embedding ────────────────────────────────────────────────────────

/// Embed all pre-compiled SPIR-V files using `include_bytes!` at compile time.
///
/// Shader names use the exact filenames (without `.spv`) from the
/// `shaders/spirv/` directory as generated by `vulkan-shaders-gen`.
fn include_all_shaders() -> std::collections::HashMap<String, Vec<u8>> {
    let mut map = std::collections::HashMap::new();

    // Helper macro: insert one shader.
    macro_rules! spv {
        ($name:literal) => {
            map.insert(
                $name.to_owned(),
                include_bytes!(concat!(env!("OUT_DIR"), "/spirv/", $name, ".spv")).to_vec(),
            );
        };
    }

    // ── Elementwise unary ────────────────────────────────────────────────
    // Only gelu_f32/tanh_f32 are actually dispatched anywhere in this
    // codebase (Rust or Python) — see gelu_tests/softcap_tests. The
    // remaining unary variants that used to be registered here
    // (silu_f32, gelu_quick_f32, relu_f32, exp_f32, sigmoid_f32, abs_f32,
    // neg_f32, ceil_f32, gelu_inplace_f32) were never dispatched anywhere
    // — confirmed via `grep -rn '"<name>"' src/ vllm_vulkan/ tests/`
    // finding matches only in scripts/compile_shaders.sh (which compiles
    // the .comp source into a .spv file regardless of whether Rust ever
    // loads it) or doc-comment examples, never a real `record_to`/
    // `execute_*` call site. See the doc comment on
    // `pipeline_cache_startup_tests` for the measured model-load-time
    // impact of removing this and the other dead registrations below.
    spv!("gelu_f32");
    spv!("tanh_f32");

    // ── Normalization / softmax / RoPE ───────────────────────────────────
    // RoPE and attention (including softmax) run on the CPU in this
    // codebase (model::cpu_rope / model::cpu_sdpa), not via GPU shaders,
    // so rms_norm_mul_rope_*, soft_max_*, and rope_* were all dead
    // registrations — same confirmation method as above.

    // ── Binary elementwise ──────────────────────────────────────────────
    // Only the f32×f32→f32 variants are dispatched (used to fuse
    // residual adds/elementwise-muls into forward_layer_gpu_matmuls's
    // post-attention GPU chain — see fused_post_attention). The f16-
    // output variants (add_f32_f32_f16, add_f32_f16_f32, mul_f32_f32_f16,
    // div_f32_f32_f16, sub_f32_f32_f16) and the ADD_RMS-fused variants
    // (add_rms_f32_f32_f32/f16, compiled from multi_add.comp — separate
    // from add.comp's own dead-code ADD_RMS branch, see mul.comp's/
    // add.comp's doc comments) were all dead registrations.
    spv!("add_f32_f32_f32");

    // ── GLU activations ─────────────────────────────────────────────────
    // swiglu_f32/geglu_f32 were dead: this model's FFN activation is
    // done as two separate dispatches (gelu_f32 then mul_f32_f32_f32),
    // not a single fused GLU shader — see forward_layer_gpu_matmuls.

    // ── Copy / reshape ──────────────────────────────────────────────────
    // get_rows_f32_f32/f16 (embedding gather), fill_f16, concat_f16, and
    // contig_cpy_f32_f16/f16_f32 were all dead: embedding lookup is done
    // via a direct CPU slice index (see forward_gpu's "Embedding"
    // section), not a GPU gather shader.

    // ── Matmul (decode: matrix-vector) ──────────────────────────────────
    // f32/f16 weights → f32 output
    spv!("mul_mat_vec_f32_f32_f32_subgroup");
    spv!("mul_mat_vec_f16_f32_f32");
    // `mul_mat_vec_f16_f32_f32_subgroup` is NOT registered here: unlike its
    // f32 counterpart (`_f32_f32_f32_subgroup`, dispatched from
    // vulkan_ops.py's fused RMSNorm→Linear path), it was never actually
    // dispatched anywhere — confirmed via `grep -rn
    // 'mul_mat_vec_f16_f32_f32_subgroup' src/ vllm_vulkan/ tests/` finding
    // zero call sites outside its own registration. See the doc comment
    // on `compile_matvec` for the measured model-load time savings from
    // removing this and the other now-unregistered dead matvec variants.
    // Quantized-weight matvec variants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/
    // Q4_K/Q5_K/Q6_K/IQ4_NL) are NOT registered here: this backend only
    // ever loads f16/f32 weights (see model_runner.py/vulkan_ops.py — no
    // quantization scheme is referenced anywhere in the Python weight-
    // loading path, and `VulkanPlatform.verify_quantization` is a no-op
    // passthrough), so these were pure dead weight — compiled into 2
    // pipelines each (base + _r4) via `compile_matvec` for zero benefit.
    // Confirmed via `grep -rn 'q4_0\|q8_0\|...' src/ vllm_vulkan/ tests/`
    // finding zero dispatch call sites for any of them. Measured
    // `PipelineCache::new()`'s (i.e. model-load) wall time before/after
    // removing them: ~1473-1474ms with vs ~733-747ms without — a
    // further ~730ms (~50%) model-load reduction on top of the earlier
    // `_r2` removal (#52), consistently reproducible across repeated
    // runs. If quantized-weight support is added later, these shaders'
    // `.comp` sources are untouched (only their `spv!`/`MATVEC_SHADERS`
    // registration is removed here) — re-registering them is a
    // one-line-per-shader change.

    // ── General matmul (prefill) ─────────────────────────────────────────
    spv!("matmul_f32_f16");
    spv!("matmul_f32_f16_aligned");
    spv!("matmul_f32_f32_fp32");
    spv!("matmul_f16_f32_fp32");

    // ── Flash attention ─────────────────────────────────────────────────
    spv!("flash_attn_f32_f16_f32");
    spv!("flash_attn_f32_f16_f16");
    spv!("flash_attn_f32_f16_f32_f16acc");
    spv!("flash_attn_f32_f16_f16_f16acc");

    // ── Misc utils ──────────────────────────────────────────────────────
    // quantize_q8_1_x4 was dead: it's a companion to the (now-removed,
    // see #53) quantized-weight matvec shaders, with no other use.

    // ── Manually compiled extras (needed for GPU inference) ──────────────
    // These were compiled directly with glslangValidator (not via vulkan-shaders-gen).
    spv!("rms_norm_f32");
    spv!("mul_f32_f32_f32");   // elementwise multiply f32×f32→f32
    // Note: rms_norm_f32_mul is created from rms_norm_f32 SPIR-V at pipeline creation
    // time by setting the do_multiply=true specialization constant (constant_id=1).
    // It is registered in the pipeline cache as "rms_norm_f32_mul" automatically.
    spv!("matmul_f32_f32");
    spv!("mul_mat_vec_f32_f32_f32");
    spv!("flash_attn_f32_f16_f32_fp32");
    spv!("paged_kv_write_f16");
    spv!("paged_kv_write_f32");
    spv!("paged_attn_decode_f16");
    spv!("paged_attn_decode_f16_coop");
    spv!("paged_attn_decode_f16_coop_512");
    spv!("paged_attn_decode_f32");
    spv!("paged_attn_decode_f32_coop");
    spv!("paged_attn_decode_f32_coop_512");

    map
}

// ─── PyO3 module ────────────────────────────────────────────────────────────

// ─── VulkanModel — end-to-end Gemma4 forward pass ────────────────────────────

/// Gemma4-E2B model with GPU-accelerated matmuls via Vulkan.
///
/// Weights are uploaded to GPU-resident host-coherent buffers once at load
/// time.  Each forward step dispatches matmuls to the GPU (one vkQueueSubmit
/// per layer) while norms and attention run on CPU.
///
/// Usage from Python:
///   vk_model = VulkanModel(safetensors_path, device_idx=0)
// Slot indices for persistent activation buffers (allocated once per model).
const ACT_QKV_IN:  usize = 0;
const ACT_Q_OUT:   usize = 1;
const ACT_K_OUT:   usize = 2;
const ACT_V_OUT:   usize = 3;
const ACT_O_IN:    usize = 4;
const ACT_O_OUT:   usize = 5;
const ACT_FFIN:    usize = 6;
const ACT_GATE:    usize = 7;
const ACT_UP:      usize = 8;
const ACT_MID:     usize = 9;
const ACT_DOWN:    usize = 10;
const ACT_GELU:    usize = 11; // gelu(gate) output           [ffn_inter]
const ACT_PLE_G:   usize = 12; // PLE gate output             [ple_dim]
const ACT_PLE_C:   usize = 13; // PLE contribution output     [H]
const ACT_PLE_LAYER: usize = 14; // PLE per-layer embed input [ple_dim]
const ACT_PLE_GELU:  usize = 15; // gelu(PLE gate) output     [ple_dim]
const ACT_PLE_MID:   usize = 16; // gelu(gate) * layer_ple    [ple_dim]
// Slots below back the fully-fused post-attention GPU chain (see
// `forward_layer_gpu_matmuls`'s `fused_post_attention` path): norms and
// residual adds that used to round-trip through the CPU between the
// o_proj / FFN / PLE submits are computed on the GPU instead, so the whole
// o_proj → ... → layer-scalar chain executes as ONE vkQueueSubmit instead
// of three. All slots below are `[H]`-sized (hidden_size), matching
// ACT_QKV_IN/ACT_O_OUT/ACT_FFIN above.
const ACT_RESIDUAL:        usize = 17; // hidden (this layer's input), for the post-attn residual add
const ACT_PA_NORMED:       usize = 18; // rms_norm_mul(o_proj_out, post_attention_layernorm.weight)
const ACT_HIDDEN2:         usize = 19; // residual + pa_normed
const ACT_FF_NORMED:       usize = 20; // rms_norm_mul(down_proj_out, post_feedforward_layernorm.weight)
const ACT_HIDDEN3A:        usize = 21; // hidden2 + ff_normed
const ACT_CONTRIB_NORMED:  usize = 22; // rms_norm_mul(ple_contrib, post_per_layer_input_norm.weight)
const ACT_HIDDEN3B:        usize = 23; // hidden3a + contrib_normed
const ACT_HIDDEN3_FINAL:   usize = 24; // hidden3b * layer_scalar — the layer's return value
// Holds this call's raw (not yet normalised) `hidden` input, so
// input_layernorm can run as a GPU dispatch (`rms_norm_f32_mul`) at the
// start of the QKV submit instead of as a separate CPU `cpu_rms_norm` call
// beforehand — see the QKV submit in `forward_layer_gpu_matmuls`.
const ACT_RAW_HIDDEN: usize = 25;
// Combined Q+K+V matvec output: `self_attn.qkv_proj.weight` (the
// concatenated Q/K/V weight rows built in `new()`) is dispatched as ONE
// matvec producing `[q_dim + 2*kv_dim]` outputs instead of three separate
// matvecs each paying their own workgroup-launch overhead — Q, K, V read
// the same input and have no dependency on each other, so this is a pure
// dispatch-count reduction, not a semantic change. ACT_Q_OUT/ACT_K_OUT/
// ACT_V_OUT above are unused by the fused QKV path (kept allocated, but
// harmlessly so — a few KB) since the CPU-fallback-adjacent code they
// once served no longer references them; removing them would mean
// renumbering every constant below, which isn't worth the risk for a
// few KB of idle memory.
const ACT_QKV_OUT: usize = 26;
// Final logit softcap (`(logits/cap).tanh()*cap`, applied over the whole
// vocab — 262144 elements for Gemma4-E2B) used to run as a single-threaded
// CPU loop after the LM head's GPU matvec, ~1.1ms every decode step
// regardless of GPU availability. Extending the LM head's existing
// vkQueueSubmit with a broadcast-mul(1/cap) -> tanh -> broadcast-mul(cap)
// chain (using the same broadcast-scalar trick as the per-layer
// `layer_scalar` multiply) measured ~2.1-2.2x faster in isolation — see
// `softcap_tests` below. ACT_INV_CAP/ACT_CAP hold the two scalar
// constants (written once, in `init_act_bufs`, since `final_logit_softcapping`
// never changes after model load); the other three are [vocab]-sized
// intermediate/final buffers for the mul/tanh/mul chain.
const ACT_LOGIT_RAW:    usize = 27;
const ACT_LOGIT_SCALED: usize = 28;
const ACT_LOGIT_TANH:   usize = 29;
const ACT_LOGIT_FINAL:  usize = 30;
const ACT_INV_CAP:      usize = 31;
const ACT_CAP:          usize = 32;
const ACT_COUNT:   usize = 33;

///   logits = vk_model.forward(token_id, position)
#[pyclass]

pub struct VulkanModel {
    inner: model::Gemma4Model,
    max_seq_len: usize,
    /// Vulkan engine for GPU matmuls (None = CPU-only fallback)
    engine: Option<compute::ComputeEngine>,
    /// GPU-resident weight buffers keyed by weight name
    gpu_weights: std::collections::HashMap<String, compute::Buffer>,
    /// Pre-allocated persistent activation buffers (fixed Vec, stable pointers)
    act_bufs: Vec<compute::Buffer>,
    /// Whether act_bufs are initialised for the current model config
    act_bufs_ready: bool,
}

#[pymethods]
impl VulkanModel {
    /// Load a Gemma4-E2B model from a safetensors file.
    ///
    /// `safetensors_path`: path to the model.safetensors file.
    /// `max_seq_len`: maximum context length (default 512).
    /// `device_idx`: Vulkan device index (default 0 = first discrete/integrated GPU).
    #[new]
    #[pyo3(signature = (safetensors_path, max_seq_len = 512, device_idx = 0))]
    fn new(safetensors_path: &str, max_seq_len: usize, device_idx: usize) -> PyResult<Self> {
        use std::path::Path;
        use std::collections::HashMap;

        let path = Path::new(safetensors_path);
        log::info!("Loading Gemma4-E2B weights from {}", safetensors_path);

        let raw_weights = model::load_weights_from_safetensors(path)
            .map_err(PyRuntimeError::new_err)?;

        log::info!("Loaded {} weight tensors, uploading to GPU...", raw_weights.len());

        let cfg = model::Gemma4Config::e2b();

        // Try to create a Vulkan compute engine for GPU-accelerated matmuls.
        let (engine_opt, gpu_weights) = match device::ComputeDevice::create(device_idx) {
            Ok(dev) => {
                // Load all pre-compiled SPIR-V shaders.
                let shader_spvs = include_all_shaders();
                let refs: HashMap<&str, &[u8]> = shader_spvs.iter()
                    .map(|(k, v)| (k.as_str(), v.as_slice())).collect();

                match compute::ComputeEngine::new(
                    dev.instance.clone(),
                    dev.physical_device,
                    dev.device.clone(),
                    dev.compute_queue,
                    dev.compute_queue_family,
                    &refs,
                ) {
                    Ok(mut engine) => {
                        // Upload all weights to GPU host-coherent buffers.
                        // Projection/matmul weights are uploaded as f16 to halve memory bandwidth.
                        // Norm/scalar weights stay as f32.
                        let mut gpu_w = HashMap::new();
                        let mut total_bytes = 0u64;
                        for (name, data) in &raw_weights {
                            let use_f16 = is_matvec_weight(name);
                            let byte_size = if use_f16 { data.len() * 2 } else { data.len() * 4 };
                            if let Ok(buf) = engine.alloc_host_coherent_storage(byte_size as u64) {
                                let dst_ptr = buf.mapped_ptr.unwrap() as *mut u8;
                                if use_f16 {
                                    for (i, &v) in data.iter().enumerate() {
                                        let h = half::f16::from_f32(v);
                                        let bits = h.to_bits().to_le_bytes();
                                        unsafe {
                                            std::ptr::copy_nonoverlapping(bits.as_ptr(), dst_ptr.add(i * 2), 2);
                                        }
                                    }
                                } else {
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            data.as_ptr() as *const u8, dst_ptr, data.len() * 4
                                        );
                                    }
                                }
                                total_bytes += byte_size as u64;
                                gpu_w.insert(name.clone(), buf);
                            }
                        }

                        // Build a concatenated Q+K+V weight per layer
                        // ("self_attn.qkv_proj.weight") so the decode hot
                        // path can dispatch one matvec producing
                        // [q_dim + 2*kv_dim] outputs instead of three
                        // separate ones. Q, K, V read the same input
                        // vector and have no data dependency on each
                        // other, so this is a pure dispatch-count
                        // reduction — concatenating the weight rows is
                        // the only change needed, since row-major [N, H]
                        // weight tensors are already contiguous, so
                        // stacking Q's rows, then K's, then V's produces
                        // exactly the [q_dim+2*kv_dim, H] matrix a single
                        // matvec needs.
                        for layer_idx in 0..cfg.num_hidden_layers {
                            let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");
                            let (Some(q), Some(k), Some(v)) = (
                                raw_weights.get(&ln("self_attn.q_proj.weight")),
                                raw_weights.get(&ln("self_attn.k_proj.weight")),
                                raw_weights.get(&ln("self_attn.v_proj.weight")),
                            ) else { continue };
                            // Encode each of Q/K/V's f32 rows directly into
                            // its slice of one f16 byte buffer — avoids
                            // materialising a combined f32 Vec (~30MB per
                            // full-attention layer, over 1GB across all 35
                            // layers) purely as a stepping stone to the f16
                            // bytes actually uploaded.
                            let mut bytes = vec![0u8; (q.len() + k.len() + v.len()) * 2];
                            let (q_bytes, rest) = bytes.split_at_mut(q.len() * 2);
                            let (k_bytes, v_bytes) = rest.split_at_mut(k.len() * 2);
                            append_f16_bytes(q_bytes, q);
                            append_f16_bytes(k_bytes, k);
                            append_f16_bytes(v_bytes, v);
                            if let Ok(buf) = engine.alloc_host_coherent_storage(bytes.len() as u64) {
                                if buf.write(&bytes).is_ok() {
                                    total_bytes += bytes.len() as u64;
                                    gpu_w.insert(ln("self_attn.qkv_proj.weight"), buf);
                                }
                            }
                        }

                        log::info!("Uploaded {:.1}GB of weights to GPU (projection weights as f16, fused QKV)",
                                   total_bytes as f64 / 1e9);
                        (Some(engine), gpu_w)
                    }
                    Err(e) => {
                        log::warn!("Failed to create compute engine: {e}, using CPU");
                        (None, HashMap::new())
                    }
                }
            }
            Err(e) => {
                log::warn!("No Vulkan device at {device_idx}: {e}, using CPU");
                (None, HashMap::new())
            }
        };

        // Keep weights also in CPU memory for the reference path.
        let weights = model::Gemma4Weights {
            tensors: raw_weights.into_iter().map(|(name, data)| {
                (name, model::SimpleTensor { data, shape: vec![] })
            }).collect(),
        };

        let kv_caches: Vec<model::KvCache> = (0..cfg.num_hidden_layers).map(|i| {
            let head_dim = cfg.layer_head_dim(i);
            model::KvCache::new(max_seq_len, cfg.num_key_value_heads, head_dim)
        }).collect();

        if engine_opt.is_some() {
            log::info!("VulkanModel ready: GPU matmuls enabled");
        } else {
            log::info!("VulkanModel ready: CPU-only mode");
        }

        Ok(VulkanModel {
            inner: model::Gemma4Model { config: cfg, weights, kv_caches },
            max_seq_len,
            engine: engine_opt,
            gpu_weights,
            act_bufs: Vec::new(),
            act_bufs_ready: false,
        })
    }

    /// Run one decode step.
    ///
    /// Returns the full logit vector [vocab_size] as a Python list of floats.
    fn forward(&mut self, token_id: u32, position: usize) -> PyResult<Vec<f32>> {
        if self.engine.is_some() {
            Ok(self.forward_gpu(token_id, position))
        } else {
            Ok(self.inner.forward(token_id, position))
        }
    }

    /// Run one decode step and sample the next token id, without ever
    /// converting the (`vocab_size`-element, 262144 for Gemma4-E2B) logit
    /// vector into a Python object.
    ///
    /// `vllm_vulkan/server.py` (the standalone Rust-`VulkanModel`-backed
    /// serving path this crate documents as giving "~3 tok/s on GB10")
    /// previously called `forward()` and passed its `Vec<f32>` return
    /// value — converted by PyO3 into 262144 individual Python `float`
    /// objects — into a pure-Python `temperature_sample()` that then did a
    /// full `sorted()` over all of them (plus several more full-vocab list
    /// comprehensions) just to pick one token. Doing the whole thing here
    /// instead means the logits never leave Rust at all: no per-element
    /// Python object conversion, and a single compiled `sort_unstable_by`
    /// instead of CPython's interpreted `sorted()` — see
    /// `model::sample_with_temperature`'s doc comment for the full
    /// rationale and `sample_logits` for a standalone-callable version of
    /// just the sampling step.
    ///
    /// `uniform_random` should be a fresh uniform `[0, 1)` draw per call
    /// (e.g. Python's `random.random()`) — see `sample_logits`.
    #[pyo3(signature = (token_id, position, temperature=1.0, top_p=1.0, top_k=64, uniform_random=0.0))]
    fn forward_and_sample(
        &mut self, token_id: u32, position: usize,
        temperature: f32, top_p: f32, top_k: i64, uniform_random: f32,
    ) -> PyResult<usize> {
        let logits = if self.engine.is_some() {
            self.forward_gpu(token_id, position)
        } else {
            self.inner.forward(token_id, position)
        };
        Ok(model::sample_with_temperature(&logits, temperature, top_p, top_k, uniform_random))
    }

    /// Reset the KV cache (start a new sequence).
    fn reset_kv_cache(&mut self) {
        for cache in self.inner.kv_caches.iter_mut() {
            cache.seq_len = 0;
        }
    }

    /// Current sequence length.
    fn seq_len(&self) -> usize {
        self.inner.kv_caches[0].seq_len
    }

    /// Number of layers.
    fn num_layers(&self) -> usize {
        self.inner.config.num_hidden_layers
    }

    /// Whether GPU acceleration is active.
    fn has_gpu(&self) -> bool {
        self.engine.is_some()
    }

    /// DEBUG: Run forward pass and return hidden state after N layers.
    /// Used to find numerical bugs by comparing with HuggingFace.
    fn forward_n_layers(&mut self, token_id: u32, position: usize, n_layers: usize) -> PyResult<Vec<f32>> {
        use model::*;
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let total_ple = cfg.num_hidden_layers * ple_dim;

        // Embedding
        let embed_w = self.inner.weights.f32_slice("model.embed_tokens.weight");
        let mut hidden: Vec<f32> = embed_w[token_id as usize * h..(token_id as usize + 1) * h]
            .iter().map(|&v| v * cfg.embed_scale).collect();

        // PLE preprocessing
        let ple_embed_w = self.inner.weights.f32_slice("model.embed_tokens_per_layer.weight");
        let ple_embeds: Vec<f32> = ple_embed_w[token_id as usize * total_ple..(token_id as usize + 1) * total_ple]
            .iter().map(|&v| v * cfg.ple_scale).collect();
        let proj_w = self.inner.weights.f32_slice("model.per_layer_model_projection.weight");
        let ple_proj = cpu_matmul(&hidden, proj_w, 1, h, total_ple);
        let ple_proj: Vec<f32> = ple_proj.iter().map(|&v| v * cfg.per_layer_projection_scale).collect();
        let pn_w = self.inner.weights.f32_slice("model.per_layer_projection_norm.weight");
        let ple_proj_normed = cpu_rms_norm(&ple_proj, pn_w, eps);
        let ple_inputs: Vec<f32> = ple_proj_normed.iter().zip(ple_embeds.iter())
            .map(|(&p, &e)| (p + e) * cfg.per_layer_input_scale).collect();

        // N layers
        for layer_idx in 0..n_layers.min(cfg.num_hidden_layers) {
            let layer_ple = &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim];
            hidden = self.inner.forward_layer(layer_idx, &hidden, position, layer_ple);
        }

        Ok(hidden)
    }
}

// Non-PyO3 implementation methods for VulkanModel
impl VulkanModel {
    /// GPU-accelerated forward pass: matmuls on GPU, norms + attention on CPU.
    fn forward_gpu(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let _t_total = std::time::Instant::now();
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let total_ple = cfg.num_hidden_layers * ple_dim;

        // ── Embedding ────────────────────────────────────────────────────────
        let embed_w = self.inner.weights.f32_slice("model.embed_tokens.weight");
        let mut hidden: Vec<f32> = embed_w[token_id as usize * h..
                                            (token_id as usize + 1) * h]
            .iter().map(|&v| v * cfg.embed_scale).collect();

        // ── PLE preprocessing ─────────────────────────────────────────────────
        let ple_embed_w = self.inner.weights.f32_slice("model.embed_tokens_per_layer.weight");
        let ple_embeds_flat: Vec<f32> = ple_embed_w[token_id as usize * total_ple..
                                                       (token_id as usize + 1) * total_ple]
            .iter().map(|&v| v * cfg.ple_scale).collect();

        // [1, H] x [total_ple, H]^T (H=1536, total_ple=8960 for Gemma4-E2B) —
        // unlike every per-layer projection (q/k/v/o_proj/gate/up/down),
        // this one previously always ran on the CPU via cpu_matmul
        // (matrixmultiply::sgemm), regardless of GPU availability. Measured
        // in isolation that's ~6.9ms/call — confirmed as real FLOPs, not a
        // GEMM-packing-overhead artifact, since a naive dot-product loop is
        // even slower — while the same mul_mat_vec_f16_f32_f32_r4 GPU
        // dispatch already used everywhere else in this file takes ~1.0-1.1ms
        // for this shape, a ~6.3-6.6x win (see ple_proj_tests below).
        // Stack-allocated + bytemuck-cast rather than a heap Vec<u8> built
        // via std::io::Write::write_all — same little-endian byte layout on
        // every platform this crate targets (x86_64/aarch64), no allocation
        // on this once-per-decode-step hot path.
        let pc_vals: [u32; 13] = [
            h as u32, h as u32, h as u32, total_ple as u32,
            (h * total_ple) as u32, h as u32, total_ple as u32,
            0u32, 0u32, 1u32, 1u32, 1u32, 1u32,
        ];
        let pc: &[u8] = bytemuck::cast_slice(&pc_vals);
        let ple_proj = self.gpu_matmul_or_cpu(
            "model.per_layer_model_projection.weight", &hidden, 1, h, total_ple, pc,
        );
        let ple_proj: Vec<f32> = ple_proj.iter()
            .map(|&v| v * cfg.per_layer_projection_scale).collect();
        let pn_w = self.inner.weights.f32_slice("model.per_layer_projection_norm.weight");
        let ple_proj_normed = model::cpu_rms_norm(&ple_proj, pn_w, eps);
        let ple_inputs: Vec<f32> = ple_proj_normed.iter().zip(ple_embeds_flat.iter())
            .map(|(&p, &e)| (p + e) * cfg.per_layer_input_scale).collect();

        if log::log_enabled!(log::Level::Debug) {
            log::debug!("PROFILE embed+ple: {}us", _t_total.elapsed().as_micros());
        }

        // ── 35 Decoder Layers ────────────────────────────────────────────────
        for layer_idx in 0..cfg.num_hidden_layers {
            let _t_layer_all = (layer_idx == 0).then(std::time::Instant::now);
            let layer_ple = &ple_inputs[layer_idx * ple_dim..(layer_idx + 1) * ple_dim];
            hidden = self.forward_layer_gpu_matmuls(layer_idx, &hidden, position, layer_ple);
            if let Some(t) = _t_layer_all {
                log::debug!("L0 total (incl. PLE): {}us", t.elapsed().as_micros());
            }
        }
        if log::log_enabled!(log::Level::Debug) {
            log::debug!("PROFILE after layers: {}us", _t_total.elapsed().as_micros());
        }

        // ── Final norm + LM head (GPU) ────────────────────────────────────────
        // LM head: [1, H] @ [vocab, H]^T — the biggest matmul, must be on GPU.
        // The final `model.norm` RMSNorm over the last hidden state ([H] =
        // 1536 elements for Gemma4-E2B) previously ran as a CPU
        // `cpu_rms_norm` call before this dispatch chain even started, then
        // its output got copied into a GPU buffer for the LM head matvec
        // anyway. `model.norm.weight` is already uploaded to `gpu_weights`
        // (every raw safetensors tensor is — see `new()`), exactly like
        // `input_layernorm.weight` is for the QKV submit, so this now runs
        // as the first dispatch of the same submit as the LM head matvec
        // and softcap chain instead, reusing ACT_RAW_HIDDEN (see its doc
        // comment above) the same way the QKV submit does. The final
        // logit softcap ((logits/cap).tanh()*cap) — previously a
        // single-threaded CPU loop over the entire vocab, ~1.1ms every
        // decode step regardless of GPU availability — already runs as
        // part of this same submit too: broadcast-mul(1/cap) -> tanh ->
        // broadcast-mul(cap), using the persistent ACT_LOGIT_*/ACT_*_CAP
        // buffers (see ACT_LOGIT_RAW's doc comment). Measured ~2.1-2.2x
        // faster than the CPU loop in isolation (softcap_tests below).
        let vocab = cfg.vocab_size;
        // Requires init_act_bufs() to have actually succeeded — not just
        // self.engine.is_some() — before touching any ACT_* slot below.
        // forward_layer_gpu_matmuls's own use_gpu already calls
        // init_act_bufs() too (it's idempotent: returns true immediately
        // once act_bufs_ready is set), so in the common case this is a
        // cheap re-check; but if every layer's use_gpu happened to be
        // false (e.g. missing per-layer weights) while embed_tokens.weight
        // is still present, act_bufs could otherwise still be empty here,
        // and every act_ptr*() call below would index out of bounds.
        let use_gpu_lm_head = self.engine.is_some()
            && self.gpu_weights.contains_key("model.embed_tokens.weight")
            && self.gpu_weights.contains_key("model.norm.weight")
            && self.init_act_bufs();
        let logits = if use_gpu_lm_head {
            let lm_w_ptr = &self.gpu_weights["model.embed_tokens.weight"] as *const compute::Buffer;
            let norm_w_ptr = &self.gpu_weights["model.norm.weight"] as *const compute::Buffer;
            self.act_bufs[ACT_RAW_HIDDEN].write(bytemuck::cast_slice(&hidden)).unwrap();

            let raw_hidden_p = self.act_ptr(ACT_RAW_HIDDEN);
            let inp_p = self.act_ptr(ACT_QKV_IN); // reuse - we're done with layer ops
            let logit_raw_p = self.act_ptr(ACT_LOGIT_RAW);
            let scaled_p = self.act_ptr(ACT_LOGIT_SCALED);
            let tanh_p = self.act_ptr(ACT_LOGIT_TANH);
            let final_p = self.act_ptr(ACT_LOGIT_FINAL);
            let inv_cap_p = self.act_ptr(ACT_INV_CAP);
            let cap_p = self.act_ptr(ACT_CAP);

            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, "rms_norm_f32_mul", &[&*raw_hidden_p, &*norm_w_ptr, &*inp_p], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_mat_vec_f32_f32_f32_r4",
                    &[&*lm_w_ptr, &*inp_p, &*logit_raw_p],
                    bytemuck::cast_slice(&mv_pc(h, vocab, 1)), (wg_r4(vocab), 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_f32_f32_f32", &[&*logit_raw_p, &*inv_cap_p, &*scaled_p], bytemuck::cast_slice(&binary_broadcast_pc(vocab)), (wg256(vocab), 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "tanh_f32", &[&*scaled_p, &*tanh_p], bytemuck::cast_slice(&unary_head_pc(vocab)), (wg128(vocab), 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_f32_f32_f32", &[&*tanh_p, &*cap_p, &*final_p], bytemuck::cast_slice(&binary_broadcast_pc(vocab)), (wg256(vocab), 1, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            read_f32_buf(unsafe { &*final_p }, vocab)
        } else {
            let norm_w = self.inner.weights.f32_slice("model.norm.weight");
            let normed = model::cpu_rms_norm(&hidden, norm_w, eps);
            let cap = cfg.final_logit_softcapping;
            let lm_w = self.inner.weights.f32_slice("model.embed_tokens.weight");
            let mut raw = model::cpu_matmul(&normed, lm_w, 1, h, vocab);
            raw.iter_mut().for_each(|l| *l = (*l / cap).tanh() * cap);
            raw
        };

        log::debug!("PROFILE total forward: {}us", _t_total.elapsed().as_micros());
        logits
    }

    /// One decoder layer: norms on CPU, matmuls on GPU via execute_batch.
    fn forward_layer_gpu_matmuls(
        &mut self,
        layer_idx: usize,
        hidden: &[f32],
        pos: usize,
        layer_ple: &[f32],
    ) -> Vec<f32> {
        let cfg = self.inner.config.clone();
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let is_full = cfg.is_full_attention(layer_idx);
        let head_dim = cfg.layer_head_dim(layer_idx);
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.num_key_value_heads;
        let q_dim = num_q * head_dim;
        let kv_dim = num_kv * head_dim;
        let is_kv_shared = cfg.is_kv_shared(layer_idx);
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let ple_dim = cfg.hidden_size_per_layer_input;
        let t = 1usize;

        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

        // Helper: pack matvec push constants
        // Pre-extract all needed weight slices as raw-pointer handles (avoids
        // borrow conflicts with the later `&mut self` GPU calls below without
        // paying for a heap allocation + memcpy on every single decode step).
        // SAFETY: `self.inner.weights` is never mutated for the lifetime of
        // `self`, so the underlying `Vec<f32>` backing storage never moves or
        // is freed while `self` is alive — these pointers stay valid for as
        // long as the `RawSlice` values derived from them are in scope here.
        macro_rules! w {
            ($name:expr) => {{
                let s = self.inner.weights.f32_slice(&ln($name));
                RawSlice { ptr: s.as_ptr(), len: s.len() }
            }};
        }

        let inln_w   = w!("input_layernorm.weight");
        let q_norm_w = w!("self_attn.q_norm.weight");
        let k_norm_w = if !is_kv_shared { Some(w!("self_attn.k_norm.weight")) } else { None };
        let pa_w     = w!("post_attention_layernorm.weight");
        let pf_w     = w!("pre_feedforward_layernorm.weight");
        let postff_w = w!("post_feedforward_layernorm.weight");
        // gate_ple_w and ple_proj_w now dispatched via GPU
        let ple_norm_w = w!("post_per_layer_input_norm.weight");
        let layer_scalar = w!("layer_scalar")[0];

        // ── ATTENTION ──────────────────────────────────────────────────────
        // `hidden` is an immutable `&[f32]` for this whole call (the caller
        // only reassigns its own `hidden` binding *after* this function
        // returns — see forward_gpu's layer loop) and `residual` below is
        // only ever read (never mutated), so it can simply alias `hidden`
        // instead of heap-allocating a full copy of it (1536 floats / 6KB)
        // on every one of the 35 layers, every decode step.
        let residual = hidden;
        let _t_layer = std::time::Instant::now();

        let shader = "mul_mat_vec_f16_f32_f32_r4";

        // Init persistent activation buffers on first call.
        let use_gpu = self.engine.is_some()
            && self.gpu_weights.contains_key(&ln("self_attn.q_proj.weight"))
            && self.gpu_weights.contains_key(&ln("input_layernorm.weight"))
            && self.init_act_bufs();

        // Whether the concatenated Q+K+V weight built in `new()` is
        // resident on the GPU, letting the QKV submit dispatch one matvec
        // instead of three (see ACT_QKV_OUT's doc comment). Falls back to
        // the three-separate-dispatch path below if not (e.g. a partial
        // upload failure) rather than failing outright.
        let use_fused_qkv = use_gpu
            && self.gpu_weights.contains_key(&ln("self_attn.qkv_proj.weight"));

        // Whether every weight the fully-fused post-attention GPU chain
        // needs (see `fused_post_attention` below) is resident on the GPU.
        // Requires strictly more weights than `use_gpu` above (that only
        // checks q_proj), since the fused path folds o_proj, the whole FFN,
        // the whole PLE branch, and every norm/residual in between into one
        // vkQueueSubmit — if any single piece is CPU-only (e.g. a partial
        // upload failure), we fall back to the old per-branch GPU/CPU
        // dispatch below rather than trying to partially fuse.
        let use_fused_post_attn = use_gpu
            && self.gpu_weights.contains_key(&ln("self_attn.o_proj.weight"))
            && self.gpu_weights.contains_key(&ln("post_attention_layernorm.weight"))
            && self.gpu_weights.contains_key(&ln("pre_feedforward_layernorm.weight"))
            && self.gpu_weights.contains_key(&ln("mlp.gate_proj.weight"))
            && self.gpu_weights.contains_key(&ln("mlp.up_proj.weight"))
            && self.gpu_weights.contains_key(&ln("mlp.down_proj.weight"))
            && self.gpu_weights.contains_key(&ln("post_feedforward_layernorm.weight"))
            && self.gpu_weights.contains_key(&ln("per_layer_input_gate.weight"))
            && self.gpu_weights.contains_key(&ln("per_layer_projection.weight"))
            && self.gpu_weights.contains_key(&ln("post_per_layer_input_norm.weight"))
            && self.gpu_weights.contains_key(&ln("layer_scalar"));

        // ── GPU BATCH: ALL 7 MATMULS IN ONE vkQueueSubmit ───────────────────
        // We batch QKV + o_proj + gate + up + down into a single command buffer.
        // The fence wait happens ONCE per layer instead of 4 times.
        // Between QKV submit and down_proj, CPU runs: Q/K/V norms, RoPE, SDPA.
        // We split into 2 submits at the attention boundary:
        //   Submit 1: Q + K + V  (before attention)
        //   Submit 2: o_proj + gate + up + down  (after attention, combined)

        let (q_vec, k_vec, v_vec) = if use_gpu {
            // Write the *raw* (not yet normalised) input to a persistent
            // buffer; input_layernorm now runs on the GPU as the first
            // dispatch in this same command buffer (see below) instead of
            // as a separate CPU `cpu_rms_norm` call before it. That removes
            // one CPU-side allocation + O(h) compute from the hot path
            // without adding a submit — it's simply the first of the four
            // dispatches already going into this one vkQueueSubmit.
            self.act_bufs[ACT_RAW_HIDDEN].write(bytemuck::cast_slice(hidden)).unwrap();

            let raw_p = self.act_ptr(ACT_RAW_HIDDEN);
            let inp = self.act_ptr(ACT_QKV_IN);
            let inln_w_gpu = &self.gpu_weights[&ln("input_layernorm.weight")] as *const compute::Buffer;

            // SUBMIT 1: input_layernorm, then Q, K, V (Q/K/V independent — no
            // barrier needed between them, only after the norm they all read).
            //
            // When the fused Q+K+V weight is available, Q/K/V collapse into
            // ONE matvec dispatch producing [q_dim+2*kv_dim] outputs (see
            // ACT_QKV_OUT's doc comment) instead of three separate ones.
            let (q_v, k_v, v_v) = if use_fused_qkv {
                let qkv_p = self.act_ptr(ACT_QKV_OUT);
                let qkv_w = &self.gpu_weights[&ln("self_attn.qkv_proj.weight")] as *const compute::Buffer;
                let qkv_dim = q_dim + 2 * kv_dim;

                let eng = self.engine.as_mut().unwrap();
                let cb = eng.begin_batch().unwrap();
                unsafe {
                    eng.record_to(cb, "rms_norm_f32_mul", &[&*raw_p, &*inln_w_gpu, &*inp], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
                    eng.record_barrier_to(cb);
                    eng.record_to(cb, shader, &[&*qkv_w, &*inp, &*qkv_p], bytemuck::cast_slice(&mv_pc(h, qkv_dim, t)), (wg_r4(qkv_dim), t as u32, 1)).unwrap();
                }
                eng.submit_batch(cb).unwrap();  // Fence wait 1: input_layernorm + fused QKV
                if layer_idx == 0 { log::debug!("L{layer_idx} QKV submit: {}µs", _t_layer.elapsed().as_micros()); }

                // The offset slicing below (`combined[..q_dim]`, etc.)
                // assumes a single token's worth of output laid out as
                // [Q | K | V] with no interleaving across tokens — true
                // for t==1 (the only value forward_layer_gpu_matmuls is
                // ever called with today: this is the single-token decode
                // path), but silently wrong for t>1, where the fused
                // matvec's output layout would need to be re-derived from
                // mv_pc's batch striding instead of assumed. Fail loudly
                // rather than risk a silent correctness bug if that ever
                // changes.
                assert_eq!(t, 1, "fused QKV output splitting assumes a single-token batch (t=1)");
                let qkv_buf_ref = unsafe { &*qkv_p };
                let q_v = read_f32_buf_at(qkv_buf_ref, 0, q_dim);
                let k_v = read_f32_buf_at(qkv_buf_ref, q_dim, kv_dim);
                let v_v = read_f32_buf_at(qkv_buf_ref, q_dim + kv_dim, kv_dim);
                (q_v, k_v, v_v)
            } else {
                let q_p = self.act_ptr(ACT_Q_OUT);
                let k_p = self.act_ptr(ACT_K_OUT);
                let v_p = self.act_ptr(ACT_V_OUT);
                let q_w = &self.gpu_weights[&ln("self_attn.q_proj.weight")] as *const compute::Buffer;
                let k_w = &self.gpu_weights[&ln("self_attn.k_proj.weight")] as *const compute::Buffer;
                let v_w = &self.gpu_weights[&ln("self_attn.v_proj.weight")] as *const compute::Buffer;

                let eng = self.engine.as_mut().unwrap();
                let cb = eng.begin_batch().unwrap();
                unsafe {
                    eng.record_to(cb, "rms_norm_f32_mul", &[&*raw_p, &*inln_w_gpu, &*inp], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
                    eng.record_barrier_to(cb);
                    eng.record_to(cb, shader, &[&*q_w, &*inp, &*q_p], bytemuck::cast_slice(&mv_pc(h, q_dim, t)), (wg_r4(q_dim), t as u32, 1)).unwrap();
                    eng.record_to(cb, shader, &[&*k_w, &*inp, &*k_p], bytemuck::cast_slice(&mv_pc(h, kv_dim, t)), (wg_r4(kv_dim), t as u32, 1)).unwrap();
                    eng.record_to(cb, shader, &[&*v_w, &*inp, &*v_p], bytemuck::cast_slice(&mv_pc(h, kv_dim, t)), (wg_r4(kv_dim), t as u32, 1)).unwrap();
                }
                eng.submit_batch(cb).unwrap();  // Fence wait 1: input_layernorm + QKV
                if layer_idx == 0 { log::debug!("L{layer_idx} QKV submit: {}µs", _t_layer.elapsed().as_micros()); }

                let q_v = read_f32_buf(unsafe { &*q_p }, t * q_dim);
                let k_v = read_f32_buf(unsafe { &*k_p }, t * kv_dim);
                let v_v = read_f32_buf(unsafe { &*v_p }, t * kv_dim);
                (q_v, k_v, v_v)
            };
            (q_v, k_v, v_v)
        } else {
            let x = model::cpu_rms_norm(hidden, &inln_w, eps);
            let q_w = self.inner.weights.f32_slice(&ln("self_attn.q_proj.weight"));
            let k_w = self.inner.weights.f32_slice(&ln("self_attn.k_proj.weight"));
            let v_w = self.inner.weights.f32_slice(&ln("self_attn.v_proj.weight"));
            (
                model::cpu_matmul(&x, q_w, 1, h, q_dim),
                model::cpu_matmul(&x, k_w, 1, h, kv_dim),
                model::cpu_matmul(&x, v_w, 1, h, kv_dim),
            )
        };

        let mut q = q_vec;
        let mut k_final = k_vec;
        let mut v_final = v_vec;

        // CPU: Q-norm, K-norm, V-norm (using pre-extracted weights), in place —
        // no per-head allocation + copy-back (see cpu_rms_norm_inplace doc).
        for hi in 0..num_q {
            let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
            model::cpu_rms_norm_inplace(s, &q_norm_w, eps);
        }
        if !is_kv_shared {
            let k_norm = k_norm_w.as_ref().unwrap();
            for hi in 0..num_kv {
                let s = &mut k_final[hi * head_dim..(hi + 1) * head_dim];
                model::cpu_rms_norm_inplace(s, k_norm, eps);
            }
            for hi in 0..num_kv {
                let s = &mut v_final[hi * head_dim..(hi + 1) * head_dim];
                model::cpu_rms_norm_no_weight_inplace(s, head_dim, eps);
            }
        }

        // CPU: RoPE
        let (theta, rotary_dim) = if is_full {
            (1_000_000.0f32, head_dim / 4)
        } else {
            (10_000.0f32, head_dim)
        };
        model::cpu_rope(&mut q, &mut k_final, pos, num_q, num_kv, head_dim, rotary_dim, theta);

        // CPU: KV cache update + SDPA
        let target_cache_idx = if is_kv_shared {
            self.inner.kv_shared_target(layer_idx)
        } else {
            self.inner.kv_caches[layer_idx].append(&k_final, &v_final);
            layer_idx
        };
        let window = if is_full { None } else { Some(cfg.sliding_window) };
        let cache = &self.inner.kv_caches[target_cache_idx];
        let attn_out = model::cpu_sdpa(
            &q, cache.k_up_to_now(), cache.v_up_to_now(),
            num_q, num_kv, head_dim, cache.seq_len, 1.0, window,
        );

        if use_fused_post_attn {
            return self.fused_post_attention(
                layer_idx, shader, &attn_out, residual, layer_ple,
                h, q_dim, ffn_inter, ple_dim, eps, t, &_t_layer,
            );
        }

        // GPU: o_proj — use persistent buffers (no alloc/free overhead)
        let o_proj = if use_gpu && self.gpu_weights.contains_key(&ln("self_attn.o_proj.weight")) {
            let attnb: &[u8] = bytemuck::cast_slice(&attn_out);
            unsafe { (*self.act_ptr_mut(ACT_O_IN)).write(attnb).unwrap(); }
            let oi   = self.act_ptr(ACT_O_IN);
            let oo   = self.act_ptr(ACT_O_OUT);
            let ow   = &self.gpu_weights[&ln("self_attn.o_proj.weight")] as *const compute::Buffer;
            let eng  = self.engine.as_mut().unwrap();
            let cb   = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, shader, &[&*ow, &*oi, &*oo], bytemuck::cast_slice(&mv_pc(q_dim, h, t)), (wg_r4(h), t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            if layer_idx == 0 { log::debug!("L{layer_idx} o_proj submit: {}µs total since layer start", _t_layer.elapsed().as_micros()); }
            FBuf::Borrowed(RawSlice { ptr: self.act_ptr_f32(ACT_O_OUT), len: t * h })
        } else {
            let ow = self.inner.weights.f32_slice(&ln("self_attn.o_proj.weight"));
            FBuf::Owned(model::cpu_matmul(&attn_out, ow, 1, q_dim, h))
        };

        // CPU: post_attn_norm + residual (using pre-extracted weight)
        let pa_normed = model::cpu_rms_norm(&o_proj, &pa_w, eps);
        let hidden2: Vec<f32> = residual.iter().zip(pa_normed.iter())
            .map(|(&r, &a)| r + a).collect();
        // hidden2 is only read from here on (never mutated), so residual2
        // can borrow it instead of cloning — this whole branch is the
        // older 3-submit CPU-fallback path (use_fused_post_attn == false),
        // kept for the CPU-only build/test configuration.
        let residual2 = &hidden2;

        // CPU: pre_ffn_norm
        let ff_in = model::cpu_rms_norm(&hidden2, &pf_w, eps);

        // SUBMIT 2 (FUSED FFN): gate + up + gelu(gate) + gelu×up + down in ONE command buffer.
        // This eliminates 3 separate submits + CPU gelu/multiply overhead.
        // Pipeline: gate_proj → [barrier] → gelu(gate) → [barrier] → gelu×up → [barrier] → down_proj
        //           up_proj runs in parallel with gate_proj (no barrier between them)
        let ff_out = if use_gpu
            && self.gpu_weights.contains_key(&ln("mlp.gate_proj.weight"))
            && self.gpu_weights.contains_key(&ln("mlp.down_proj.weight"))
        {
            let ffb: &[u8] = bytemuck::cast_slice(&ff_in);
            unsafe { (*self.act_ptr_mut(ACT_FFIN)).write(ffb).unwrap(); }

            let ffi    = self.act_ptr(ACT_FFIN);
            let gp     = self.act_ptr(ACT_GATE);   // gate_proj output (src for gelu)
            let gelu_p = self.act_ptr(ACT_GELU);   // gelu(gate) output (dst from gelu, src for mul)
            let up_p   = self.act_ptr(ACT_UP);
            let mid_p  = self.act_ptr(ACT_MID);
            let down_p = self.act_ptr(ACT_DOWN);

            let gw  = &self.gpu_weights[&ln("mlp.gate_proj.weight")] as *const compute::Buffer;
            let uw  = &self.gpu_weights[&ln("mlp.up_proj.weight")]   as *const compute::Buffer;
            let dw  = &self.gpu_weights[&ln("mlp.down_proj.weight")] as *const compute::Buffer;

            // Push constants for elementwise ops over ffn_inter elements
            // gelu_f32: local_size_x=128, mul_f32_f32_f32: local_size_x=256
            let gelu_wg = ((ffn_inter + 127) / 128) as u32;
            let mul_wg  = ((ffn_inter + 255) / 256) as u32;
            let ew_wg = gelu_wg; // used below for gelu dispatch

            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                // Step 1: gate and up matmuls are independent (same input ffi, different outputs)
                eng.record_to(cb, shader, &[&*gw, &*ffi, &*gp], bytemuck::cast_slice(&mv_pc(h, ffn_inter, t)), (wg_r4(ffn_inter), t as u32, 1)).unwrap();
                eng.record_to(cb, shader, &[&*uw, &*ffi, &*up_p], bytemuck::cast_slice(&mv_pc(h, ffn_inter, t)), (wg_r4(ffn_inter), t as u32, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 2: gelu(gate) → gelu_p  (gelu_f32: binding0=src, binding1=dst)
                eng.record_to(cb, "gelu_f32", &[&*gp, &*gelu_p], bytemuck::cast_slice(&unary_head_pc(ffn_inter)), (ew_wg, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 3: mid = gelu(gate) * up
                eng.record_to(cb, "mul_f32_f32_f32", &[&*gelu_p, &*up_p, &*mid_p], bytemuck::cast_slice(&binary_elementwise_pc(ffn_inter)), (mul_wg, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 4: ff_out = down_proj(mid)
                eng.record_to(cb, shader, &[&*dw, &*mid_p, &*down_p], bytemuck::cast_slice(&mv_pc(ffn_inter, h, t)), (wg_r4(h), t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();  // ONE fence wait for all FFN ops
            if layer_idx == 0 { log::debug!("L{layer_idx} FFN submit: {}µs total since layer start", _t_layer.elapsed().as_micros()); }

            FBuf::Borrowed(RawSlice { ptr: self.act_ptr_f32(ACT_DOWN), len: t * h })
        } else {
            // CPU fallback
            let gate_w = self.inner.weights.f32_slice(&ln("mlp.gate_proj.weight")).to_vec();
            let up_w   = self.inner.weights.f32_slice(&ln("mlp.up_proj.weight")).to_vec();
            let gate = model::cpu_matmul(&ff_in, &gate_w, 1, h, ffn_inter);
            let up   = model::cpu_matmul(&ff_in, &up_w,   1, h, ffn_inter);
            let gate_act = model::cpu_gelu(&gate);
            let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
            FBuf::Owned(self.gpu_matmul_or_cpu(&ln("mlp.down_proj.weight"), &mid, t, ffn_inter, h, bytemuck::cast_slice(&mv_pc(ffn_inter, h, t))))
        };

        // CPU: post_ffn_norm + residual (using pre-extracted weight)
        let ff_normed = model::cpu_rms_norm(&ff_out, &postff_w, eps);
        let mut hidden3: Vec<f32> = residual2.iter().zip(ff_normed.iter())
            .map(|(&r, &f)| r + f).collect();

        // PLE (per-layer embedding) contribution: gate_proj → gelu → ×layer_ple → proj,
        // all four steps fused into ONE command buffer / vkQueueSubmit (mirrors the
        // FFN fusion above). Previously this was 2 separate submits with a CPU
        // gelu + elementwise-multiply round trip in between; fusing it into a
        // single GPU submit removes one fence-wait per layer from the decode step.
        let contrib = if use_gpu
            && self.gpu_weights.contains_key(&ln("per_layer_input_gate.weight"))
            && self.gpu_weights.contains_key(&ln("per_layer_projection.weight"))
        {
            let h3b: &[u8] = bytemuck::cast_slice(&hidden3);
            unsafe { (*self.act_ptr_mut(ACT_FFIN)).write(h3b).unwrap(); }  // reuse ACT_FFIN as PLE input
            let lpb: &[u8] = bytemuck::cast_slice(layer_ple);
            unsafe { (*self.act_ptr_mut(ACT_PLE_LAYER)).write(lpb).unwrap(); }

            let inp_p   = self.act_ptr(ACT_FFIN);
            let pg_p    = self.act_ptr(ACT_PLE_G);
            let gelu_p  = self.act_ptr(ACT_PLE_GELU);
            let layer_p = self.act_ptr(ACT_PLE_LAYER);
            let mid_p   = self.act_ptr(ACT_PLE_MID);
            let pc_p    = self.act_ptr(ACT_PLE_C);

            let pgw = &self.gpu_weights[&ln("per_layer_input_gate.weight")] as *const compute::Buffer;
            let ppw = &self.gpu_weights[&ln("per_layer_projection.weight")] as *const compute::Buffer;

            let gelu_wg_ple = ((ple_dim + 127) / 128) as u32;
            let mul_wg_ple  = ((ple_dim + 255) / 256) as u32;

            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, shader, &[&*pgw, &*inp_p, &*pg_p], bytemuck::cast_slice(&mv_pc(h, ple_dim, t)), (wg_r4(ple_dim), t as u32, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "gelu_f32", &[&*pg_p, &*gelu_p], bytemuck::cast_slice(&unary_head_pc(ple_dim)), (gelu_wg_ple, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_f32_f32_f32", &[&*gelu_p, &*layer_p, &*mid_p], bytemuck::cast_slice(&binary_elementwise_pc(ple_dim)), (mul_wg_ple, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, shader, &[&*ppw, &*mid_p, &*pc_p], bytemuck::cast_slice(&mv_pc(ple_dim, h, t)), (wg_r4(h), t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();  // ONE fence wait for the whole PLE branch
            FBuf::Borrowed(RawSlice { ptr: self.act_ptr_f32(ACT_PLE_C), len: t * h })
        } else {
            let pgw = self.inner.weights.f32_slice(&ln("per_layer_input_gate.weight"));
            let gate_ple = model::cpu_matmul(&hidden3, pgw, 1, h, ple_dim);
            let gate_ple_act = model::cpu_gelu(&gate_ple);
            let gated: Vec<f32> = gate_ple_act.iter().zip(layer_ple.iter())
                .map(|(&g, &p)| g * p).collect();
            let ppw = self.inner.weights.f32_slice(&ln("per_layer_projection.weight"));
            FBuf::Owned(model::cpu_matmul(&gated, ppw, 1, ple_dim, h))
        };
        let contrib_normed = model::cpu_rms_norm(&contrib, &ple_norm_w, eps);
        hidden3.iter_mut().zip(contrib_normed.iter()).for_each(|(hv, &c)| *hv += c);

        // Layer scalar (pre-extracted)
        hidden3.iter_mut().for_each(|v| *v *= layer_scalar);
        if layer_idx == 0 { log::debug!("L{layer_idx} END: {}µs total", _t_layer.elapsed().as_micros()); }
        hidden3
    }

    /// The whole post-attention half of a decoder layer — o_proj plus its
    /// norm and residual, the FFN (gate/up/gelu/mul/down) plus its norm
    /// and residual, the PLE branch (gate/gelu/mul/proj) plus its norm
    /// and residual, and the final layer-scalar multiply — as ONE command
    /// buffer / vkQueueSubmit, instead of the three separate submits
    /// (o_proj, FFN, PLE) the `use_gpu`-but-not-`use_fused_post_attn` path
    /// below still uses, with a CPU round trip for every norm/residual add
    /// in between.
    ///
    /// Every fence wait measured on real Vulkan hardware in this repo's CI
    /// costs far more than the handful of extra elementwise dispatches
    /// this adds inside a single submit (barriers between dispatches in
    /// the *same* command buffer are cheap; a `vkQueueSubmit` + fence wait
    /// is not) — see the perf test in `matvec_fusion_tests` below for a
    /// same-hardware, same-shapes A/B measurement of exactly this trade-off.
    #[allow(clippy::too_many_arguments)]
    fn fused_post_attention(
        &mut self,
        layer_idx: usize,
        shader: &str,
        attn_out: &[f32],
        residual: &[f32],
        layer_ple: &[f32],
        h: usize,
        q_dim: usize,
        ffn_inter: usize,
        ple_dim: usize,
        eps: f32,
        t: usize,
        t_layer: &std::time::Instant,
    ) -> Vec<f32> {
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

        // Upload this call's fresh inputs to their persistent buffers. No
        // outstanding borrow on `self` yet at this point, so plain safe
        // indexing works — no need for the raw-pointer/unsafe pattern the
        // rest of this function uses to work around holding `self.engine`
        // mutably borrowed at the same time as `self.act_bufs`/`self.gpu_weights`.
        self.act_bufs[ACT_O_IN].write(bytemuck::cast_slice(attn_out)).unwrap();
        self.act_bufs[ACT_RESIDUAL].write(bytemuck::cast_slice(residual)).unwrap();
        self.act_bufs[ACT_PLE_LAYER].write(bytemuck::cast_slice(layer_ple)).unwrap();

        let oi               = self.act_ptr(ACT_O_IN);
        let oo               = self.act_ptr(ACT_O_OUT);
        let res_p            = self.act_ptr(ACT_RESIDUAL);
        let pa_normed_p      = self.act_ptr(ACT_PA_NORMED);
        let hidden2_p        = self.act_ptr(ACT_HIDDEN2);
        let ffi              = self.act_ptr(ACT_FFIN); // reused as ff_in, like the non-fused path
        let gp               = self.act_ptr(ACT_GATE);
        let gelu_p           = self.act_ptr(ACT_GELU);
        let up_p             = self.act_ptr(ACT_UP);
        let mid_p            = self.act_ptr(ACT_MID);
        let down_p           = self.act_ptr(ACT_DOWN);
        let ff_normed_p      = self.act_ptr(ACT_FF_NORMED);
        let hidden3a_p       = self.act_ptr(ACT_HIDDEN3A);
        let pg_p             = self.act_ptr(ACT_PLE_G);
        let ple_gelu_p       = self.act_ptr(ACT_PLE_GELU);
        let layer_p          = self.act_ptr(ACT_PLE_LAYER);
        let ple_mid_p        = self.act_ptr(ACT_PLE_MID);
        let pc_p             = self.act_ptr(ACT_PLE_C);
        let contrib_normed_p = self.act_ptr(ACT_CONTRIB_NORMED);
        let hidden3b_p       = self.act_ptr(ACT_HIDDEN3B);
        let hidden3_final_p  = self.act_ptr(ACT_HIDDEN3_FINAL);

        let ow             = &self.gpu_weights[&ln("self_attn.o_proj.weight")] as *const compute::Buffer;
        let pa_w_gpu       = &self.gpu_weights[&ln("post_attention_layernorm.weight")] as *const compute::Buffer;
        let pf_w_gpu       = &self.gpu_weights[&ln("pre_feedforward_layernorm.weight")] as *const compute::Buffer;
        let gw             = &self.gpu_weights[&ln("mlp.gate_proj.weight")] as *const compute::Buffer;
        let uw             = &self.gpu_weights[&ln("mlp.up_proj.weight")] as *const compute::Buffer;
        let dw             = &self.gpu_weights[&ln("mlp.down_proj.weight")] as *const compute::Buffer;
        let postff_w_gpu   = &self.gpu_weights[&ln("post_feedforward_layernorm.weight")] as *const compute::Buffer;
        let pgw            = &self.gpu_weights[&ln("per_layer_input_gate.weight")] as *const compute::Buffer;
        let ppw            = &self.gpu_weights[&ln("per_layer_projection.weight")] as *const compute::Buffer;
        let ple_norm_w_gpu = &self.gpu_weights[&ln("post_per_layer_input_norm.weight")] as *const compute::Buffer;
        let layer_scalar_gpu = &self.gpu_weights[&ln("layer_scalar")] as *const compute::Buffer;

        let eng = self.engine.as_mut().unwrap();
        let cb = eng.begin_batch().unwrap();
        unsafe {
            // o_proj
            eng.record_to(cb, shader, &[&*ow, &*oi, &*oo], bytemuck::cast_slice(&mv_pc(q_dim, h, t)), (wg_r4(h), t as u32, 1)).unwrap();
            eng.record_barrier_to(cb);
            // post_attn_norm (weight-multiplying RMSNorm) + residual add
            eng.record_to(cb, "rms_norm_f32_mul", &[&*oo, &*pa_w_gpu, &*pa_normed_p], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "add_f32_f32_f32", &[&*res_p, &*pa_normed_p, &*hidden2_p], bytemuck::cast_slice(&binary_elementwise_pc(h)), (wg256(h), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            // pre_ffn_norm
            eng.record_to(cb, "rms_norm_f32_mul", &[&*hidden2_p, &*pf_w_gpu, &*ffi], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            // FFN: gate + up (independent) -> gelu(gate) -> gelu*up -> down_proj
            eng.record_to(cb, shader, &[&*gw, &*ffi, &*gp], bytemuck::cast_slice(&mv_pc(h, ffn_inter, t)), (wg_r4(ffn_inter), t as u32, 1)).unwrap();
            eng.record_to(cb, shader, &[&*uw, &*ffi, &*up_p], bytemuck::cast_slice(&mv_pc(h, ffn_inter, t)), (wg_r4(ffn_inter), t as u32, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "gelu_f32", &[&*gp, &*gelu_p], bytemuck::cast_slice(&unary_head_pc(ffn_inter)), (wg128(ffn_inter), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "mul_f32_f32_f32", &[&*gelu_p, &*up_p, &*mid_p], bytemuck::cast_slice(&binary_elementwise_pc(ffn_inter)), (wg256(ffn_inter), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, shader, &[&*dw, &*mid_p, &*down_p], bytemuck::cast_slice(&mv_pc(ffn_inter, h, t)), (wg_r4(h), t as u32, 1)).unwrap();
            eng.record_barrier_to(cb);
            // post_ffn_norm + residual add
            eng.record_to(cb, "rms_norm_f32_mul", &[&*down_p, &*postff_w_gpu, &*ff_normed_p], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "add_f32_f32_f32", &[&*hidden2_p, &*ff_normed_p, &*hidden3a_p], bytemuck::cast_slice(&binary_elementwise_pc(h)), (wg256(h), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            // PLE: gate -> gelu -> ×layer_ple -> proj
            eng.record_to(cb, shader, &[&*pgw, &*hidden3a_p, &*pg_p], bytemuck::cast_slice(&mv_pc(h, ple_dim, t)), (wg_r4(ple_dim), t as u32, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "gelu_f32", &[&*pg_p, &*ple_gelu_p], bytemuck::cast_slice(&unary_head_pc(ple_dim)), (wg128(ple_dim), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "mul_f32_f32_f32", &[&*ple_gelu_p, &*layer_p, &*ple_mid_p], bytemuck::cast_slice(&binary_elementwise_pc(ple_dim)), (wg256(ple_dim), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, shader, &[&*ppw, &*ple_mid_p, &*pc_p], bytemuck::cast_slice(&mv_pc(ple_dim, h, t)), (wg_r4(h), t as u32, 1)).unwrap();
            eng.record_barrier_to(cb);
            // contrib norm + residual add
            eng.record_to(cb, "rms_norm_f32_mul", &[&*pc_p, &*ple_norm_w_gpu, &*contrib_normed_p], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "add_f32_f32_f32", &[&*hidden3a_p, &*contrib_normed_p, &*hidden3b_p], bytemuck::cast_slice(&binary_elementwise_pc(h)), (wg256(h), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            // Final layer-scalar multiply (broadcast a single scalar weight).
            eng.record_to(cb, "mul_f32_f32_f32", &[&*hidden3b_p, &*layer_scalar_gpu, &*hidden3_final_p], bytemuck::cast_slice(&binary_broadcast_pc(h)), (wg256(h), 1, 1)).unwrap();
        }
        eng.submit_batch(cb).unwrap(); // ONE fence wait for the entire post-attention chain
        if layer_idx == 0 {
            log::debug!("L{layer_idx} fused post-attn submit: {}µs total since layer start", t_layer.elapsed().as_micros());
        }
        // `eng`'s mutable borrow of `self.engine` ended at `submit_batch`
        // above, so the result can be read back with plain safe indexing.
        read_f32_buf(&self.act_bufs[ACT_HIDDEN3_FINAL], h)
    }

    /// GPU matmul (single) with CPU fallback.
    /// Initialise persistent activation buffers sized for one decode step.
    fn init_act_bufs(&mut self) -> bool {
        if self.act_bufs_ready { return true; }
        let eng = match self.engine.as_mut() { Some(e) => e, None => return false };
        let cfg = &self.inner.config;
        let h = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.global_head_dim; // largest (full-attn layers)
        let kv_dim = cfg.num_key_value_heads * cfg.global_head_dim;
        let ffn_inter = cfg.intermediate_size * 2; // largest (double-wide KV-shared layers)

        let ple_dim = cfg.hidden_size_per_layer_input; // 256
        let vocab = cfg.vocab_size;
        let cap = cfg.final_logit_softcapping;
        let sizes: [u64; ACT_COUNT] = [
            (h * 4) as u64,          // ACT_QKV_IN
            (q_dim * 4) as u64,      // ACT_Q_OUT
            (kv_dim * 4) as u64,     // ACT_K_OUT
            (kv_dim * 4) as u64,     // ACT_V_OUT
            (q_dim * 4) as u64,      // ACT_O_IN
            (h * 4) as u64,          // ACT_O_OUT
            (h * 4) as u64,          // ACT_FFIN
            (ffn_inter * 4) as u64,  // ACT_GATE
            (ffn_inter * 4) as u64,  // ACT_UP
            (ffn_inter * 4) as u64,  // ACT_MID
            (h * 4) as u64,          // ACT_DOWN
            (ffn_inter * 4) as u64,  // ACT_GELU
            (ple_dim * 4) as u64,    // ACT_PLE_G
            (h * 4) as u64,          // ACT_PLE_C
            (ple_dim * 4) as u64,    // ACT_PLE_LAYER
            (ple_dim * 4) as u64,    // ACT_PLE_GELU
            (ple_dim * 4) as u64,    // ACT_PLE_MID
            (h * 4) as u64,          // ACT_RESIDUAL
            (h * 4) as u64,          // ACT_PA_NORMED
            (h * 4) as u64,          // ACT_HIDDEN2
            (h * 4) as u64,          // ACT_FF_NORMED
            (h * 4) as u64,          // ACT_HIDDEN3A
            (h * 4) as u64,          // ACT_CONTRIB_NORMED
            (h * 4) as u64,          // ACT_HIDDEN3B
            (h * 4) as u64,          // ACT_HIDDEN3_FINAL
            (h * 4) as u64,          // ACT_RAW_HIDDEN
            ((q_dim + 2 * kv_dim) * 4) as u64, // ACT_QKV_OUT
            (vocab * 4) as u64,      // ACT_LOGIT_RAW
            (vocab * 4) as u64,      // ACT_LOGIT_SCALED
            (vocab * 4) as u64,      // ACT_LOGIT_TANH
            (vocab * 4) as u64,      // ACT_LOGIT_FINAL
            4,                       // ACT_INV_CAP
            4,                       // ACT_CAP
        ];

        self.act_bufs.clear();
        for &sz in &sizes {
            match eng.alloc_host_coherent_storage(sz) {
                Ok(buf) => self.act_bufs.push(buf),
                Err(e) => {
                    log::warn!("Failed to allocate activation buffer: {e}");
                    self.act_bufs.clear();
                    return false;
                }
            }
        }
        // final_logit_softcapping never changes after model load, so its
        // scalar broadcast operands only need writing once here rather
        // than on every decode step.
        self.act_bufs[ACT_INV_CAP].write(bytemuck::cast_slice(&[1.0f32 / cap])).unwrap();
        self.act_bufs[ACT_CAP].write(bytemuck::cast_slice(&[cap])).unwrap();
        self.act_bufs_ready = true;
        log::info!("Allocated {} persistent activation buffers ({:.1} KB)",
            ACT_COUNT,
            sizes.iter().sum::<u64>() as f32 / 1024.0);
        true
    }

    /// Get pointer to a persistent activation buffer by slot index.
    /// SAFETY: caller must not hold other borrows to act_bufs.
    fn act_ptr(&self, slot: usize) -> *const compute::Buffer {
        &self.act_bufs[slot] as *const compute::Buffer
    }

    fn act_ptr_mut(&mut self, slot: usize) -> *mut compute::Buffer {
        &mut self.act_bufs[slot] as *mut compute::Buffer
    }

    /// Raw pointer to a persistent activation buffer's mapped memory, viewed
    /// as `f32`. Used with `RawSlice`/`FBuf` to read a GPU dispatch's output
    /// directly out of the mapped buffer without the `Vec<f32>` allocation +
    /// copy that `read_f32_buf` does, for call sites that only need to *read*
    /// the result (see `FBuf` doc comment). Detached from `self`'s borrow —
    /// same rationale as `act_ptr`/`act_ptr_mut` above.
    fn act_ptr_f32(&self, slot: usize) -> *const f32 {
        self.act_bufs[slot].mapped_ptr.unwrap() as *const f32
    }

    fn gpu_matmul_or_cpu(&mut self, weight_name: &str, x: &[f32],
                          t: usize, k: usize, n: usize, pc: &[u8]) -> Vec<f32> {
        if let (Some(eng), Some(w_ptr)) = (
            self.engine.as_mut(),
            self.gpu_weights.get(weight_name).map(|b| b as *const compute::Buffer)
        ) {
            let xb: &[u8] = bytemuck::cast_slice(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(xb).unwrap();
            let out = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, "mul_mat_vec_f16_f32_f32_r4",
                    &[&*w_ptr, &*inp_p, &*out_p], pc, (wg_r4(n), t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let result = read_f32_buf(&out, t * n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out);
            result
        } else {
            let w = self.inner.weights.f32_slice(weight_name);
            model::cpu_matmul(x, w, t, k, n)
        }
    }

    /// GPU double-matmul (gate + up) with CPU fallback.
    fn gpu_matmul2_or_cpu(&mut self, w1_name: &str, w2_name: &str, x: &[f32],
                           t: usize, k: usize, n: usize, pc: &[u8]) -> (Vec<f32>, Vec<f32>) {
        if let (Some(eng), true) = (
            self.engine.as_mut(),
            self.gpu_weights.contains_key(w1_name) && self.gpu_weights.contains_key(w2_name)
        ) {
            let w1_ptr = &self.gpu_weights[w1_name] as *const compute::Buffer;
            let w2_ptr = &self.gpu_weights[w2_name] as *const compute::Buffer;
            let xb: &[u8] = bytemuck::cast_slice(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(xb).unwrap();
            let out1 = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let out2 = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let inp_p  = &inp  as *const compute::Buffer;
            let out1_p = &out1 as *const compute::Buffer;
            let out2_p = &out2 as *const compute::Buffer;
            let cb = eng.begin_batch().unwrap();
            unsafe {
                let sh = "mul_mat_vec_f16_f32_f32_r4";
                eng.record_to(cb, sh, &[&*w1_ptr, &*inp_p, &*out1_p], pc, (wg_r4(n), t as u32, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, sh, &[&*w2_ptr, &*inp_p, &*out2_p], pc, (wg_r4(n), t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            let r1 = read_f32_buf(&out1, t * n);
            let r2 = read_f32_buf(&out2, t * n);
            eng.return_to_pool(inp);
            eng.return_to_pool(out1);
            eng.return_to_pool(out2);
            (r1, r2)
        } else {
            let w1 = self.inner.weights.f32_slice(w1_name);
            let w2 = self.inner.weights.f32_slice(w2_name);
            (model::cpu_matmul(x, w1, t, k, n),
             model::cpu_matmul(x, w2, t, k, n))
        }
    }

    pub fn num_layers_impl(&self) -> usize {
        self.inner.config.num_hidden_layers
    }
}



// ─── Helper functions for VulkanModel ────────────────────────────────────────

/// A non-owning handle to an `[f32]` slice, identified by raw pointer + len.
///
/// Used to reference small per-layer weight tensors (norm weights, scalars)
/// across a sequence of `&mut self` calls (GPU dispatch, buffer writes) that
/// the borrow checker cannot otherwise reconcile with a live `&self.inner`
/// borrow, without resorting to a heap allocation + memcpy on every decode
/// step. `Deref<Target = [f32]>` gives it the same call-site ergonomics as
/// `Vec<f32>` (e.g. `&inln_w`, `inln_w[0]`) via deref coercion.
#[derive(Clone, Copy)]
struct RawSlice {
    ptr: *const f32,
    len: usize,
}

// SAFETY: RawSlice is only ever constructed from a `&[f32]` borrowed out of
// `VulkanModel::inner.weights`, which is immutable and pinned for the whole
// lifetime of the model. Values are used only on the thread that created
// them (Vulkan buffer handles are already !Send in this crate).
impl std::ops::Deref for RawSlice {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        // SAFETY: see struct docs — the backing Vec<f32> outlives every
        // RawSlice derived from it and is never mutated in between.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Either a borrowed activation-buffer view (`RawSlice`, GPU dispatch path)
/// or an owned `Vec<f32>` (CPU-fallback path), unified behind one type so
/// call sites like `forward_layer_gpu_matmuls`'s `let x = if use_gpu {...}
/// else {...};` — which must produce the same type from both branches — can
/// read a GPU dispatch's output directly out of its persistent, mapped
/// buffer with no allocation, instead of both branches always paying for a
/// `read_f32_buf` heap allocation + copy just to satisfy the CPU fallback's
/// `Vec<f32>` return type. Only usable where the result is read, never
/// mutated in place, after construction (RoPE/RMSNorm's in-place update
/// paths still go through owned `Vec<f32>`, since those aren't valid
/// `RawSlice` targets — mutating a `Buffer` still in flight for other reads
/// would be unsound).
enum FBuf {
    Borrowed(RawSlice),
    Owned(Vec<f32>),
}

impl std::ops::Deref for FBuf {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            FBuf::Borrowed(s) => s,
            FBuf::Owned(v) => v,
        }
    }
}

/// Returns true if this weight tensor should be uploaded to GPU as f16.
/// Norm weights, scalars, and embeddings stay as f32 for precision.
fn is_matvec_weight(name: &str) -> bool {
    // Projection weights: q/k/v/o/gate/up/down, PLE gate, PLE projection
    name.ends_with("_proj.weight")          // e.g. q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        || name.ends_with("_gate.weight")   // e.g. per_layer_input_gate.weight
        || name.ends_with("_projection.weight")  // e.g. per_layer_projection.weight
        // embed_tokens.weight stays f32 (LM head precision), layernorm/scalar stay f32
}

/// Shared "generic_binary_head" push-constant layout used by
/// `mul_mat_vec.comp`'s sibling elementwise/norm shaders (`add.comp`,
/// `mul.comp`, `rms_norm.comp`): `ne, ne00-03, nb00-03, ne10-13, nb10-13,
/// ne20-23, nb20-23, misalign, param1, param2, param3` (29 x 4 bytes — see
/// `shaders/generic_binary_head.glsl`). Every tensor `forward_layer_gpu_matmuls`
/// dispatches these against is a flat `[n]` row vector (batch/channel dims
/// are all 1, single token per decode step), so only the element counts
/// (`ne00`, `ne10`, `ne20`) and `param1` actually matter here; every other
/// field is filled with the same "contiguous 1-D tensor" convention this
/// file's `mv_pc`/`gelu_pc` closures already use elsewhere.
///
/// `b_len` is the second operand's element count: pass `n` for a plain
/// elementwise op (residual add, RMSNorm weight multiply) or `1` to
/// broadcast a single scalar against every output element (the per-layer
/// scalar multiply) — `src1_idx`'s `fastmod(i00, ne10)` is always 0 when
/// `ne10 == 1`, so every element reads `data_b[0]`.
///
/// Returns a stack-allocated `[u32; 29]` instead of a heap-allocated
/// `Vec<u8>` — every field here is fully determined by shapes that are
/// fixed for a layer's whole lifetime (`h`, `q_dim`, `ffn_inter`, ...:
/// only ~4 distinct combinations exist across all 35 of Gemma4-E2B's
/// layers), so the previous `Vec::with_capacity` + `write_all` construction
/// paid for a fresh heap allocation on every single dispatch — several
/// hundred times per decode step — for a 116-byte value that's cheaper to
/// build directly on the stack. `f32`/`i32` fields are stored via
/// `.to_bits()`/`as u32` so the whole array stays one `u32`-typed buffer
/// that `bytemuck::cast_slice` can reinterpret as `&[u8]` at the call site
/// without any unsafe code here.
fn binary_head_pc(n: usize, b_len: usize, param1: f32) -> [u32; 29] {
    let nu = n as u32;
    let bu = b_len as u32;
    [
        nu, nu, 1, 1, 1, 1, nu, nu, nu,
        bu, 1, 1, 1, 1, bu, bu, bu,
        nu, 1, 1, 1, 1, nu, nu, nu,
        0,               // misalign
        param1.to_bits(), // param1 (eps for rms_norm; 0.0 otherwise)
        0.0f32.to_bits(), // param2 (unused)
        0i32 as u32,      // param3 (unused)
    ]
}

/// Push constants for `rms_norm_f32_mul` (the weight-multiplying RMSNorm
/// variant) over a single `[n]`-element row: `param1` becomes the epsilon
/// `rms_norm.comp` reads, and the weight operand's length equals `n` so the
/// shader's broadcast check (`ncols > ne10`) takes the simple, non-broadcast
/// path.
fn rms_norm_mul_pc(n: usize, eps: f32) -> [u32; 29] {
    binary_head_pc(n, n, eps)
}

/// Push constants for `add_f32_f32_f32` / `mul_f32_f32_f32` over two
/// same-shape `[n]`-element rows (residual adds, RMSNorm-weight multiply
/// is done via `rms_norm_mul_pc` instead — this is for plain elementwise).
fn binary_elementwise_pc(n: usize) -> [u32; 29] {
    binary_head_pc(n, n, 0.0)
}

/// Push constants for `mul_f32_f32_f32` broadcasting a single scalar
/// (`layer_scalar`) against every element of an `[n]`-element row.
fn binary_broadcast_pc(n: usize) -> [u32; 29] {
    binary_head_pc(n, 1, 0.0)
}

/// Push constants for the `generic_head.glsl`-based elementwise unary
/// shaders (`gelu_f32`, `tanh_f32`, ...): `KX, KY, param1-4` (6 x 4 bytes).
/// `KX` is the element count; the other fields are unused by `tanh_f32`.
/// Stack-allocated for the same reason as `binary_head_pc` above.
fn unary_head_pc(n: usize) -> [u32; 6] {
    [n as u32, 1, 0, 0, 0, 0]
}

/// Push constants for the `mul_mat_vec_*_r4` matvec shaders: `ncols,
/// stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b,
/// batch_stride_d, fusion_flags, base_work_group_y, ne02, ne12,
/// broadcast2, broadcast3` (13 x 4 bytes) — every matvec dispatch in
/// `forward_layer_gpu_matmuls`/`fused_post_attention` builds this from
/// just `(k, n, t)`. Stack-allocated for the same reason as
/// `binary_head_pc` above: this was previously two identical closures
/// (one per function) each heap-allocating a fresh `Vec<u8>` on every
/// one of ~10 matvec dispatches per layer, per decode step.
fn mv_pc(k: usize, n: usize, t: usize) -> [u32; 13] {
    [
        k as u32, k as u32, k as u32, n as u32,
        (k * n) as u32, k as u32, n as u32,
        0, 0, 1, t as u32, t as u32, 1,
    ]
}

/// Workgroup count for `add_f32_f32_f32` / `mul_f32_f32_f32` dispatches
/// (256 threads/workgroup, up to 2 elements/thread — see `mul.comp`;
/// matches the FFN's existing `mul_wg` calculation).
fn wg256(n: usize) -> u32 {
    n.div_ceil(256) as u32
}

/// Workgroup count for `gelu_f32` and `tanh_f32` (both 128 threads/workgroup,
/// 1 element/thread — see `gelu.comp` / `tanh.comp`; both independently
/// measured to be faster at `local_size_x=128` than the more common 512
/// at their respective real dispatch sizes: `ple_dim`/`ffn_inter` for
/// gelu, `vocab` for tanh).
fn wg128(n: usize) -> u32 {
    n.div_ceil(128) as u32
}

/// Workgroup count for the `_r4` matvec shader variants
/// (`mul_mat_vec_{f16,f32}_f32_f32_r4`): each workgroup computes 4 output
/// rows instead of 1 (`NUM_ROWS=4`, vs. the base variant's implicit
/// `NUM_ROWS=1`), so a matvec with `n` output rows needs `ceil(n/4)`
/// workgroups instead of `n`. These `_r4` pipelines were already compiled
/// by `pipeline.rs`'s `compile_matvec` (for every shader in
/// `MATVEC_SHADERS`) but never dispatched anywhere in this file — cutting
/// the workgroup count by ~4x measurably speeds up every matvec dispatch
/// in the decode hot path (1.1x-1.5x across the shapes Gemma4-E2B
/// actually uses, see `matvec_r4_tests` below), apparently because
/// per-workgroup *launch* overhead, not per-element bandwidth, dominates
/// a wide-but-shallow (single-token) matvec on the Vulkan driver
/// available for testing.
fn wg_r4(n: usize) -> u32 {
    n.div_ceil(4) as u32
}

/// Convert f32 weights to f16 bytes for GPU upload.
/// f16 halves memory bandwidth which is the main bottleneck for matvec ops.
fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; data.len() * 2];
    append_f16_bytes(&mut bytes[..], data);
    bytes
}

/// Encodes `data` as little-endian f16 bytes directly into `dst` (which
/// must be exactly `data.len() * 2` bytes), with no intermediate
/// allocation. Used to build the concatenated Q+K+V weight buffer in
/// `new()` without first materialising a combined `Vec<f32>` — for
/// full-attention layers that's ~30MB of f32 Q+K+V weight per layer (over
/// 1GB across all 35 layers) that would otherwise be allocated and freed
/// purely as a stepping stone to the f16 bytes actually uploaded.
fn append_f16_bytes(dst: &mut [u8], data: &[f32]) {
    debug_assert_eq!(dst.len(), data.len() * 2);
    for (i, &v) in data.iter().enumerate() {
        let h = half::f16::from_f32(v);
        dst[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
}

/// Quantize f32 weights to Q8_0 format for GPU upload.
/// Q8_0: blocks of 32 elements, each block has a f16 scale + 32 int8 values = 34 bytes/block.
/// This halves memory vs f16 (2.2× faster matvec) at minor precision cost.
/// Requirement: data.len() must be a multiple of 32.
fn f32_to_q8_0_bytes(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0, "Q8_0 requires data length divisible by 32, got {}", data.len());
    let num_blocks = data.len() / 32;
    let mut out = vec![0u8; num_blocks * 34]; // 34 bytes per block (2 f16 + 32 int8)
    for b in 0..num_blocks {
        let block = &data[b * 32..(b + 1) * 32];
        // Find max absolute value for scale
        let absmax = block.iter().copied().fold(0.0f32, |m, v| if v.abs() > m { v.abs() } else { m });
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1e-10_f32 };
        let scale_f16 = half::f16::from_f32(scale);
        let offset = b * 34;
        out[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
        for (j, &v) in block.iter().enumerate() {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            out[offset + 2 + j] = q as u8;
        }
    }
    out
}

fn read_f32_buf(buf: &compute::Buffer, count: usize) -> Vec<f32> {
    let ptr = buf.mapped_ptr.unwrap() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, count).to_vec() }
}

/// Like `read_f32_buf`, but starting `offset` `f32` elements into the
/// buffer instead of at the start — lets a single dispatch's output
/// (e.g. the fused QKV matvec's `[Q | K | V]` layout) be split into
/// several owned `Vec<f32>`s with one allocation each, instead of reading
/// the whole thing into one `Vec<f32>` and then `.to_vec()`-ing each
/// sub-slice out of *that* (an extra, avoidable full-size copy).
fn read_f32_buf_at(buf: &compute::Buffer, offset: usize, count: usize) -> Vec<f32> {
    let ptr = buf.mapped_ptr.unwrap() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr.add(offset), count).to_vec() }
}

/// Python-visible module `vllm_vulkan._rs`.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::try_init().ok();

    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_count, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_devices, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(get_memory_info, m)?)?;
    m.add_function(wrap_pyfunction!(sample_logits, m)?)?;

    m.add_class::<VulkanDevice>()?;
    m.add_class::<VulkanContext>()?;
    m.add_class::<GpuTensor>()?;
    m.add_class::<VulkanModel>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__vulkan_available__", device::is_vulkan_available())?;

    Ok(())
}

/// Serializes every GPU-timing test in this file's `#[cfg(test)]` modules.
///
/// `cargo test` runs tests concurrently (one thread per test by default);
/// each GPU perf test here creates its own `ComputeDevice`/`ComputeEngine`
/// and submits work to the same physical GPU queue, so two such tests
/// running at once contend for the GPU and can make an otherwise
/// consistently-faster dispatch measure as slower purely from scheduling
/// noise (observed directly: `matvec_r4_tests`'s r4-vs-base comparison,
/// already merged and normally reliable, spuriously failed once under full
/// suite parallelism while investigating this — re-running it alone showed
/// the expected win every time). Every GPU perf/correctness test acquires
/// this lock for its full duration so at most one such test ever touches
/// the GPU at a time, regardless of `cargo test`'s thread count.
#[cfg(test)]
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquires `GPU_TEST_LOCK`, tolerating poisoning: the lock only ever
/// guards "don't run two GPU-timing tests at once", not any shared data,
/// so a prior test panicking while holding it (e.g. a failed assertion)
/// doesn't mean the next test needs to fail too.
#[cfg(test)]
fn gpu_test_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod matvec_fusion_tests {
    //! Validates `fused_post_attention` (the single-vkQueueSubmit
    //! o_proj→FFN→PLE→scalar chain in `forward_layer_gpu_matmuls`) against
    //! the exact same math run through individual CPU primitives, and
    //! against the older three-separate-submits GPU path it replaces —
    //! both reached through the real `forward_layer_gpu_matmuls` entry
    //! point, not a reimplementation. Requires a real Vulkan device; skips
    //! cleanly on headless CI runners with no GPU/ICD.
    use super::*;
    use std::collections::HashMap as Map;

    /// Deterministic pseudo-random f32s in [-1, 1) (xorshift64*), so the
    /// test needs no `rand` dependency and no downloaded model checkpoint.
    fn fake_random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
                let bits = state.wrapping_mul(0x2545F4914F6CDD1D);
                ((bits >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn l2_rel_err(a: &[f32], b: &[f32]) -> f32 {
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            diff_sq += ((x - y) as f64).powi(2);
            ref_sq += (x as f64).powi(2);
        }
        (diff_sq / ref_sq.max(1e-12)).sqrt() as f32
    }

    /// Builds a `VulkanModel` with real Gemma4-E2B layer-0 dimensions, but
    /// small random weights (not a downloaded checkpoint — the full e2b()
    /// vocab/embedding tables alone are multiple GiB, which this harness
    /// has no need for: it only exercises `forward_layer_gpu_matmuls`,
    /// never the embedding or LM head). `layer_scalar` is uploaded to the
    /// GPU only when `with_layer_scalar_on_gpu` is true — omitting it is
    /// how the test forces `use_fused_post_attn` to be false, to compare
    /// the new fused path against the older three-submit path through the
    /// exact same production entry point.
    // pub(crate): reused by ple_proj_tests below, which needs a real
    // VulkanModel to exercise gpu_matmul_or_cpu directly.
    pub(crate) fn build_test_model(with_layer_scalar_on_gpu: bool) -> Option<VulkanModel> {
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: no Vulkan device available ({e})"); return None; }
        };
        let shader_spvs = include_all_shaders();
        let refs: Map<&str, &[u8]> = shader_spvs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let mut engine = compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).expect("create ComputeEngine");

        let cfg = model::Gemma4Config::e2b();
        let h = cfg.hidden_size;
        let layer_idx = 0usize; // sliding-window, non-KV-shared layer
        let head_dim = cfg.layer_head_dim(layer_idx);
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let ffn_inter = cfg.layer_intermediate_size(layer_idx);
        let ple_dim = cfg.hidden_size_per_layer_input;
        let ln = |s: &str| format!("model.layers.{layer_idx}.{s}");

        // (name, seed, len) for every tensor forward_layer_gpu_matmuls needs
        // for layer 0. Shapes match Gemma4Config::e2b() exactly.
        let tensors: Vec<(String, u64, usize)> = vec![
            (ln("input_layernorm.weight"), 1, h),
            (ln("self_attn.q_proj.weight"), 2, q_dim * h),
            (ln("self_attn.k_proj.weight"), 3, kv_dim * h),
            (ln("self_attn.v_proj.weight"), 4, kv_dim * h),
            (ln("self_attn.q_norm.weight"), 5, head_dim),
            (ln("self_attn.k_norm.weight"), 6, head_dim),
            (ln("self_attn.o_proj.weight"), 7, h * q_dim),
            (ln("post_attention_layernorm.weight"), 8, h),
            (ln("pre_feedforward_layernorm.weight"), 9, h),
            (ln("mlp.gate_proj.weight"), 10, ffn_inter * h),
            (ln("mlp.up_proj.weight"), 11, ffn_inter * h),
            (ln("mlp.down_proj.weight"), 12, h * ffn_inter),
            (ln("post_feedforward_layernorm.weight"), 13, h),
            (ln("per_layer_input_gate.weight"), 14, ple_dim * h),
            (ln("per_layer_projection.weight"), 15, h * ple_dim),
            (ln("post_per_layer_input_norm.weight"), 16, h),
            (ln("layer_scalar"), 17, 1),
            // Global (not per-layer) PLE projection weight, used by
            // ple_proj_tests below to exercise gpu_matmul_or_cpu directly —
            // shape matches forward_gpu's total_ple = num_hidden_layers *
            // hidden_size_per_layer_input (35 * 256 = 8960 for e2b()).
            ("model.per_layer_model_projection.weight".to_string(), 18,
                cfg.num_hidden_layers * cfg.hidden_size_per_layer_input * h),
        ];

        let mut cpu_tensors = Map::new();
        let mut gpu_weights = Map::new();
        for (name, seed, len) in &tensors {
            let data = fake_random(*len, *seed);
            let is_matvec = is_matvec_weight(name);
            if name.ends_with("layer_scalar") && !with_layer_scalar_on_gpu {
                // Deliberately omit: forces use_fused_post_attn == false.
            } else if is_matvec {
                let bytes = f32_to_f16_bytes(&data);
                let buf = engine.alloc_host_coherent_storage(bytes.len() as u64).unwrap();
                buf.write(&bytes).unwrap();
                gpu_weights.insert(name.clone(), buf);
            } else {
                let bytes: &[u8] = bytemuck::cast_slice(&data);
                let buf = engine.alloc_host_coherent_storage(bytes.len() as u64).unwrap();
                buf.write(bytes).unwrap();
                gpu_weights.insert(name.clone(), buf);
            }
            cpu_tensors.insert(name.clone(), model::SimpleTensor { data, shape: vec![] });
        }

        // Mirror new()'s concatenated-QKV upload so tests actually exercise
        // the fused QKV dispatch path (forward_layer_gpu_matmuls's
        // use_fused_qkv), not just fall back to the three-separate-dispatch
        // path because "self_attn.qkv_proj.weight" is missing.
        {
            let q = &cpu_tensors[&ln("self_attn.q_proj.weight")].data;
            let k = &cpu_tensors[&ln("self_attn.k_proj.weight")].data;
            let v = &cpu_tensors[&ln("self_attn.v_proj.weight")].data;
            let mut combined = Vec::with_capacity(q.len() + k.len() + v.len());
            combined.extend_from_slice(q);
            combined.extend_from_slice(k);
            combined.extend_from_slice(v);
            let bytes = f32_to_f16_bytes(&combined);
            let buf = engine.alloc_host_coherent_storage(bytes.len() as u64).unwrap();
            buf.write(&bytes).unwrap();
            gpu_weights.insert(ln("self_attn.qkv_proj.weight"), buf);
        }

        let kv_caches: Vec<model::KvCache> = (0..cfg.num_hidden_layers).map(|i| {
            let hd = cfg.layer_head_dim(i);
            model::KvCache::new(64, cfg.num_key_value_heads, hd)
        }).collect();

        Some(VulkanModel {
            inner: model::Gemma4Model {
                config: cfg,
                weights: model::Gemma4Weights { tensors: cpu_tensors },
                kv_caches,
            },
            max_seq_len: 64,
            engine: Some(engine),
            gpu_weights,
            act_bufs: Vec::new(),
            act_bufs_ready: false,
        })
    }

    #[test]
    fn fused_post_attention_matches_three_submit_path() {
        let _guard = gpu_test_guard();
        let Some(mut fused_model) = build_test_model(true) else { return };
        let Some(mut old_model) = build_test_model(false) else { return };

        let hidden = fake_random(fused_model.inner.config.hidden_size, 100);
        let ple_dim = fused_model.inner.config.hidden_size_per_layer_input;
        let layer_ple = fake_random(ple_dim, 101);

        assert!(fused_model.engine.is_some());
        let out_fused = fused_model.forward_layer_gpu_matmuls(0, &hidden, 0, &layer_ple);
        let out_old = old_model.forward_layer_gpu_matmuls(0, &hidden, 0, &layer_ple);

        let err = l2_rel_err(&out_old, &out_fused);
        println!("fused vs 3-submit path: l2_rel_err={err:.6}");
        // Both paths run the identical f16-quantized matvec weights and f32
        // norm/residual math — any difference should be pure floating-point
        // reassociation noise, not a real numerical divergence.
        assert!(err < 1e-4, "fused post-attention diverged from the 3-submit path: {err}");
    }

    /// Validates the GPU `rms_norm_f32_mul` dispatch that now computes
    /// input_layernorm at the start of the QKV submit (see
    /// `forward_layer_gpu_matmuls`) against `Gemma4Model::forward_layer`,
    /// the independent pure-CPU reference implementation in src/model.rs
    /// (which still calls `cpu_rms_norm` for this step). Uses two freshly
    /// built models with independent KV caches so neither call's
    /// KV-cache-append affects the other.
    #[test]
    fn gpu_input_layernorm_matches_cpu_reference() {
        let _guard = gpu_test_guard();
        let Some(mut gpu_model) = build_test_model(true) else { return };
        let Some(mut cpu_ref_model) = build_test_model(true) else { return };

        let hidden = fake_random(gpu_model.inner.config.hidden_size, 300);
        let ple_dim = gpu_model.inner.config.hidden_size_per_layer_input;
        let layer_ple = fake_random(ple_dim, 301);

        assert!(gpu_model.engine.is_some());
        let out_gpu = gpu_model.forward_layer_gpu_matmuls(0, &hidden, 0, &layer_ple);
        let out_cpu_ref = cpu_ref_model.inner.forward_layer(0, &hidden, 0, &layer_ple);

        let err = l2_rel_err(&out_cpu_ref, &out_gpu);
        println!("GPU input_layernorm+QKV path vs pure-CPU reference: l2_rel_err={err:.6}");
        // The GPU path quantizes matvec weights to f16 (the pure-CPU
        // reference keeps everything f32), so some divergence is expected
        // — this is the same tolerance already established for the
        // existing GPU-vs-CPU comparisons in this file.
        assert!(err < 0.01, "GPU input_layernorm diverged from the CPU reference: {err}");
    }

    #[test]
    fn fused_post_attention_is_faster_than_three_submit_path() {
        let _guard = gpu_test_guard();
        let Some(mut fused_model) = build_test_model(true) else { return };
        let Some(mut old_model) = build_test_model(false) else { return };

        let hidden = fake_random(fused_model.inner.config.hidden_size, 200);
        let ple_dim = fused_model.inner.config.hidden_size_per_layer_input;
        let layer_ple = fake_random(ple_dim, 201);

        // Warm up (pipeline creation, first-dispatch driver overhead).
        for _ in 0..5 {
            fused_model.forward_layer_gpu_matmuls(0, &hidden, 0, &layer_ple);
            old_model.forward_layer_gpu_matmuls(0, &hidden, 0, &layer_ple);
        }

        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            fused_model.forward_layer_gpu_matmuls(0, &hidden, 0, &layer_ple);
        }
        let fused_elapsed = t0.elapsed();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            old_model.forward_layer_gpu_matmuls(0, &hidden, 0, &layer_ple);
        }
        let old_elapsed = t0.elapsed();

        let fused_us = fused_elapsed.as_micros() as f64 / iters as f64;
        let old_us = old_elapsed.as_micros() as f64 / iters as f64;
        println!(
            "forward_layer_gpu_matmuls: 3-submit {old_us:.1}us/call   fused-1-submit {fused_us:.1}us/call   speedup {:.2}x",
            old_us / fused_us
        );
        assert!(
            fused_us < old_us,
            "fused post-attention ({fused_us:.1}us) was not faster than the 3-submit path ({old_us:.1}us)"
        );
    }
}

#[cfg(test)]
mod matvec_r4_tests {
    //! Validates the `_r4` (`NUM_ROWS=4`) matvec pipeline variants now used
    //! throughout `forward_layer_gpu_matmuls`/`fused_post_attention`/the LM
    //! head (see `wg_r4`'s doc comment) against the base (`NUM_ROWS=1`)
    //! variant they replace: same SPIR-V module, same math, just a
    //! different `NUM_ROWS` specialization constant and correspondingly
    //! fewer/bigger workgroups. Requires a real Vulkan device; skips
    //! cleanly (not a failure) on headless CI runners with no GPU/ICD.
    use super::*;

    fn fake_random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
                let bits = state.wrapping_mul(0x2545F4914F6CDD1D);
                ((bits >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn l2_rel_err(a: &[f32], b: &[f32]) -> f32 {
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            diff_sq += ((x - y) as f64).powi(2);
            ref_sq += (x as f64).powi(2);
        }
        (diff_sq / ref_sq.max(1e-12)).sqrt() as f32
    }

    fn matvec_pc(k: usize, n: usize) -> Vec<u8> {
        use std::io::Write;
        let mut v = Vec::with_capacity(13 * 4);
        for x in [k as u32, k as u32, k as u32, n as u32,
                   (k * n) as u32, k as u32, n as u32,
                   0u32, 0u32, 1u32, 1u32, 1u32, 1u32] {
            v.write_all(&x.to_le_bytes()).unwrap();
        }
        v
    }

    struct Harness { engine: compute::ComputeEngine }

    fn make_harness() -> Option<Harness> {
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: no Vulkan device available ({e})"); return None; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let engine = compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).expect("create ComputeEngine");
        Some(Harness { engine })
    }

    fn dispatch(h: &mut Harness, shader: &str, weight: &compute::Buffer, x: &[f32], k: usize, n: usize, wg: u32) -> Vec<f32> {
        let eng = &mut h.engine;
        let xb: &[u8] = bytemuck::cast_slice(x);
        let inp = eng.alloc_host_coherent_storage(xb.len() as u64).unwrap();
        inp.write(xb).unwrap();
        let out = eng.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        let pc = matvec_pc(k, n);
        let cb = eng.begin_batch().unwrap();
        eng.record_to(cb, shader, &[weight, &inp, &out], &pc, (wg, 1, 1)).unwrap();
        eng.submit_batch(cb).unwrap();
        read_f32_buf(&out, n)
    }

    fn check_shape(h: &mut Harness, label: &str, k: usize, n: usize) {
        let weight = fake_random(n * k, 1);
        let x = fake_random(k, 2);
        let f16_bytes = f32_to_f16_bytes(&weight);
        let buf = h.engine.alloc_host_coherent_storage(f16_bytes.len() as u64).unwrap();
        buf.write(&f16_bytes).unwrap();

        let base = dispatch(h, "mul_mat_vec_f16_f32_f32", &buf, &x, k, n, n as u32);
        let r4 = dispatch(h, "mul_mat_vec_f16_f32_f32_r4", &buf, &x, k, n, wg_r4(n));

        let err = l2_rel_err(&base, &r4);
        println!("{label:<22} k={k:>6} n={n:>6}  base-vs-r4 l2_rel_err={err:.6}");
        assert!(err < 1e-5, "{label}: _r4 output diverged from the base variant: {err}");
    }

    #[test]
    fn r4_matches_base_at_gemma4_e2b_shapes() {
        let _guard = gpu_test_guard();
        let Some(mut h) = make_harness() else { return };
        // Every matvec shape forward_layer_gpu_matmuls/fused_post_attention
        // actually dispatches (src/model.rs Gemma4Config::e2b()).
        check_shape(&mut h, "q_proj (sliding)", 1536, 2048);
        check_shape(&mut h, "q_proj (full-attn)", 1536, 4096);
        check_shape(&mut h, "o_proj (sliding)", 2048, 1536);
        check_shape(&mut h, "gate/up_proj", 1536, 6144);
        check_shape(&mut h, "down_proj (KV-shared)", 12288, 1536);
        check_shape(&mut h, "ple_gate/proj", 1536, 256);
    }

    #[test]
    fn r4_is_faster_than_base_at_gemma4_e2b_shapes() {
        let _guard = gpu_test_guard();
        let Some(mut h) = make_harness() else { return };

        let k = 1536usize;
        let n = 6144usize; // gate/up_proj width — the widest single-weight
                            // dispatch outside the KV-shared FFN/LM head.
        let weight = fake_random(n * k, 3);
        let x = fake_random(k, 4);
        let f16_bytes = f32_to_f16_bytes(&weight);
        let buf = h.engine.alloc_host_coherent_storage(f16_bytes.len() as u64).unwrap();
        buf.write(&f16_bytes).unwrap();

        for _ in 0..5 {
            dispatch(&mut h, "mul_mat_vec_f16_f32_f32", &buf, &x, k, n, n as u32);
            dispatch(&mut h, "mul_mat_vec_f16_f32_f32_r4", &buf, &x, k, n, wg_r4(n));
        }

        let iters = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { dispatch(&mut h, "mul_mat_vec_f16_f32_f32", &buf, &x, k, n, n as u32); }
        let base_elapsed = t0.elapsed();

        let t0 = std::time::Instant::now();
        for _ in 0..iters { dispatch(&mut h, "mul_mat_vec_f16_f32_f32_r4", &buf, &x, k, n, wg_r4(n)); }
        let r4_elapsed = t0.elapsed();

        let base_us = base_elapsed.as_micros() as f64 / iters as f64;
        let r4_us = r4_elapsed.as_micros() as f64 / iters as f64;
        println!(
            "matvec [1,{k}] x [{n},{k}]^T  base(1-row/wg) {base_us:.1}us/call   _r4(4-rows/wg) {r4_us:.1}us/call   speedup {:.2}x",
            base_us / r4_us
        );
        assert!(
            r4_us < base_us,
            "_r4 ({r4_us:.1}us) was not faster than the base variant ({base_us:.1}us)"
        );
    }
}

#[cfg(test)]
mod gelu_tests {
    //! Validates `gelu_f32`'s GPU output against `model::cpu_gelu` at the
    //! real Gemma4-E2B shapes the shader is actually dispatched at:
    //! `ple_dim` (256 — where `gelu.comp`'s `local_size_x=128` tuning is
    //! a measurable win over the more common 512, since 512 would leave
    //! 3/4 of a single workgroup's lanes idle at this size) and
    //! `ffn_inter` (6144 / 12288 — where 128 measured as a tie with 512,
    //! no regression). Requires a real Vulkan device; skips cleanly (not
    //! a failure) on headless CI runners with no GPU/ICD.
    use super::*;

    fn make_engine() -> Option<compute::ComputeEngine> {
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: no Vulkan device available ({e})"); return None; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        Some(compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).expect("create ComputeEngine"))
    }

    #[test]
    fn gpu_gelu_matches_cpu_reference() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        for &n in &[256usize, 6144, 12288] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 3.0).collect();
            let cpu_result = model::cpu_gelu(&x);

            let inp = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            inp.write(bytemuck::cast_slice(&x)).unwrap();
            let out = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();

            let cb = engine.begin_batch().unwrap();
            engine.record_to(cb, "gelu_f32", &[&inp, &out], bytemuck::cast_slice(&unary_head_pc(n)), (wg128(n), 1, 1)).unwrap();
            engine.submit_batch(cb).unwrap();
            let gpu_result = read_f32_buf(&out, n);

            let mut max_err = 0.0f32;
            for (&a, &b) in cpu_result.iter().zip(gpu_result.iter()) {
                max_err = max_err.max((a - b).abs());
            }
            assert!(max_err < 1e-4, "n={n}: GPU gelu_f32 (local_size_x=128) diverged from cpu_gelu: max abs err={max_err}");
        }
    }
}

#[cfg(test)]
mod binary_elementwise_tests {
    //! Validates `mul_f32_f32_f32` / `add_f32_f32_f32`'s GPU output
    //! against a CPU reference at the real Gemma4-E2B shapes they're
    //! actually dispatched at (`ple_dim`=256, `hidden_size`=1536,
    //! `ffn_inter`=6144/12288, `vocab`=262144), after removing their
    //! redundant `num_iter=2` second iteration (see mul.comp/add.comp's
    //! `main()` doc comments for the full coverage-math argument for why
    //! `num_iter=1` is sufficient — each workgroup's `num_iter=2` second
    //! iteration wrote exactly the same range the next workgroup's first
    //! iteration wrote, so it was pure redundant GPU work, not additional
    //! coverage). Requires a real Vulkan device; skips cleanly (not a
    //! failure) on headless CI runners with no GPU/ICD.
    use super::*;

    fn make_engine() -> Option<compute::ComputeEngine> {
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: no Vulkan device available ({e})"); return None; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        Some(compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).expect("create ComputeEngine"))
    }

    #[test]
    fn mul_matches_cpu_reference() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        for &n in &[256usize, 1536, 6144, 12288, 262144] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
            let cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

            let ab = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            ab.write(bytemuck::cast_slice(&a)).unwrap();
            let bb = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            bb.write(bytemuck::cast_slice(&b)).unwrap();
            let out = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();

            let cb = engine.begin_batch().unwrap();
            engine.record_to(cb, "mul_f32_f32_f32", &[&ab, &bb, &out], bytemuck::cast_slice(&binary_elementwise_pc(n)), (wg256(n), 1, 1)).unwrap();
            engine.submit_batch(cb).unwrap();
            let gpu_result = read_f32_buf(&out, n);

            let mut max_err = 0.0f32;
            for (&x, &y) in cpu_result.iter().zip(gpu_result.iter()) {
                max_err = max_err.max((x - y).abs());
            }
            assert!(max_err < 1e-5, "n={n}: GPU mul_f32_f32_f32 (num_iter=1) diverged from CPU reference: max abs err={max_err}");
        }
    }

    #[test]
    fn add_matches_cpu_reference() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        for &n in &[256usize, 1536, 6144, 12288, 262144] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).cos()).collect();
            let cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

            let ab = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            ab.write(bytemuck::cast_slice(&a)).unwrap();
            let bb = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
            bb.write(bytemuck::cast_slice(&b)).unwrap();
            let out = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();

            let cb = engine.begin_batch().unwrap();
            engine.record_to(cb, "add_f32_f32_f32", &[&ab, &bb, &out], bytemuck::cast_slice(&binary_elementwise_pc(n)), (wg256(n), 1, 1)).unwrap();
            engine.submit_batch(cb).unwrap();
            let gpu_result = read_f32_buf(&out, n);

            let mut max_err = 0.0f32;
            for (&x, &y) in cpu_result.iter().zip(gpu_result.iter()) {
                max_err = max_err.max((x - y).abs());
            }
            assert!(max_err < 1e-5, "n={n}: GPU add_f32_f32_f32 (num_iter=1) diverged from CPU reference: max abs err={max_err}");
        }
    }
}

#[cfg(test)]
mod pipeline_cache_startup_tests {
    //! `compile_matvec` used to also compile a NUM_ROWS=2 (`_r2`) variant
    //! for all 17 MATVEC_SHADERS entries, never actually dispatched
    //! anywhere in this codebase (confirmed via `grep -rn '_r2' src/
    //! vllm_vulkan/ tests/` finding zero dispatch call sites — every
    //! real matvec dispatch uses either the unsuffixed base variant, the
    //! `_subgroup` unsuffixed variant, or `_r4`). Measured directly
    //! (`PipelineCache::new()`, i.e. model-load, timed before/after
    //! removing it): ~1865-1885ms with `_r2` vs ~1461-1467ms without,
    //! consistently reproducible across repeated runs — a ~400ms
    //! (~22%) model-load startup-time win from removing this dead
    //! pipeline (see #52). The same audit later found the 13 quantized-
    //! weight matvec variants (#53, ~730ms/~50% further reduction) and
    //! `mul_mat_vec_f16_f32_f32_subgroup` (below) were equally dead —
    //! unlike its f32 counterpart (`_f32_f32_f32_subgroup`, dispatched
    //! from vulkan_ops.py's fused RMSNorm→Linear path), the f16 variant
    //! was never actually dispatched anywhere. These tests are
    //! deterministic (non-flaky) structural regression guards confirming
    //! specific variants are no longer compiled, rather than wall-clock
    //! assertions (which would be sensitive to the specific CI runner's
    //! absolute performance).
    use super::*;

    fn make_engine() -> Option<compute::ComputeEngine> {
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: no Vulkan device available ({e})"); return None; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        Some(compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).expect("create ComputeEngine"))
    }

    #[test]
    fn r2_matvec_variant_is_no_longer_compiled() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        // Trivial zero-filled data: this test only checks whether
        // `record_to` finds a compiled pipeline for each shader name
        // (Ok) or not (Err "not found") — the actual numerical result is
        // irrelevant here, unlike the correctness tests elsewhere.
        let k = 64usize;
        let n = 4usize;
        let weight_f16 = f32_to_f16_bytes(&vec![0.0f32; n * k]);
        let x = vec![0.0f32; k];
        let wbuf = engine.alloc_host_coherent_storage(weight_f16.len() as u64).unwrap();
        wbuf.write(&weight_f16).unwrap();
        let xbuf = engine.alloc_host_coherent_storage((k * 4) as u64).unwrap();
        xbuf.write(bytemuck::cast_slice(&x)).unwrap();
        let out = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        let pc = mv_pc(k, n, 1);

        // The base and _r4 variants must still be present and dispatchable...
        let cb = engine.begin_batch().unwrap();
        let res_base = engine.record_to(
            cb, "mul_mat_vec_f16_f32_f32", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n as u32, 1, 1),
        );
        assert!(res_base.is_ok(), "base matvec variant should still be compiled: {res_base:?}");

        let res_r4 = engine.record_to(
            cb, "mul_mat_vec_f16_f32_f32_r4", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (wg_r4(n), 1, 1),
        );
        assert!(res_r4.is_ok(), "_r4 matvec variant should still be compiled: {res_r4:?}");

        // ...but the dead _r2 variant must be gone.
        let res_r2 = engine.record_to(
            cb, "mul_mat_vec_f16_f32_f32_r2", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n.div_ceil(2) as u32, 1, 1),
        );
        assert!(
            res_r2.is_err(),
            "expected '_r2' matvec variant to no longer be compiled (dead code, never \
             dispatched anywhere), but record_to succeeded: {res_r2:?}"
        );
    }

    #[test]
    fn quantized_matvec_variants_are_no_longer_compiled() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        // Trivial zero-filled data — see r2_matvec_variant_is_no_longer_compiled
        // above for why numerical correctness is irrelevant to this test.
        // Sized generously (n*k*4 bytes) to be large enough for both the f16
        // (n*k*2 bytes) and f32 (n*k*4 bytes) shader variants dispatched
        // below — this test never submits the batch, but an undersized
        // buffer would still be technically incorrect (out-of-bounds reads
        // if ever submitted, or under strict validation layers).
        let k = 64usize;
        let n = 4usize;
        let weight = vec![0u8; n * k * 4];
        let x = vec![0.0f32; k];
        let wbuf = engine.alloc_host_coherent_storage(weight.len() as u64).unwrap();
        wbuf.write(&weight).unwrap();
        let xbuf = engine.alloc_host_coherent_storage((k * 4) as u64).unwrap();
        xbuf.write(bytemuck::cast_slice(&x)).unwrap();
        let out = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        let pc = mv_pc(k, n, 1);

        let cb = engine.begin_batch().unwrap();
        for name in [
            "mul_mat_vec_q4_0_f32_f32",
            "mul_mat_vec_q4_0_f32_f32_subgroup",
            "mul_mat_vec_q4_1_f32_f32",
            "mul_mat_vec_q5_0_f32_f32",
            "mul_mat_vec_q5_1_f32_f32",
            "mul_mat_vec_q8_0_f32_f32",
            "mul_mat_vec_q8_0_f32_f32_subgroup",
            "mul_mat_vec_q2_k_f16_f32",
            "mul_mat_vec_q3_k_f16_f32",
            "mul_mat_vec_q4_k_f32_f32_subgroup",
            "mul_mat_vec_q5_k_f16_f32",
            "mul_mat_vec_q6_k_f32_f32_subgroup",
            "mul_mat_vec_iq4_nl_f32_f32",
        ] {
            let res = engine.record_to(cb, name, &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n as u32, 1, 1));
            assert!(
                res.is_err(),
                "expected quantized matvec variant '{name}' to no longer be compiled \
                 (dead code, never dispatched anywhere — this backend only ever loads \
                 f16/f32 weights), but record_to succeeded: {res:?}"
            );
        }

        // The f32/f16 variants actually used in production must still work.
        let res_f32 = engine.record_to(cb, "mul_mat_vec_f32_f32_f32", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n as u32, 1, 1));
        assert!(res_f32.is_ok(), "mul_mat_vec_f32_f32_f32 should still be compiled: {res_f32:?}");
        let res_f16 = engine.record_to(cb, "mul_mat_vec_f16_f32_f32", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n as u32, 1, 1));
        assert!(res_f16.is_ok(), "mul_mat_vec_f16_f32_f32 should still be compiled: {res_f16:?}");
    }

    #[test]
    fn f16_subgroup_matvec_variant_is_no_longer_compiled() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        // Trivial zero-filled data — see r2_matvec_variant_is_no_longer_compiled
        // above for why numerical correctness is irrelevant to this test.
        // Sized generously (n*k*4 bytes) to be large enough for both the f16
        // (n*k*2 bytes) and f32 (n*k*4 bytes) shader variants dispatched
        // below — this test never submits the batch, but an undersized
        // buffer would still be technically incorrect (out-of-bounds reads
        // if ever submitted, or under strict validation layers).
        let k = 64usize;
        let n = 4usize;
        let weight = vec![0u8; n * k * 4];
        let x = vec![0.0f32; k];
        let wbuf = engine.alloc_host_coherent_storage(weight.len() as u64).unwrap();
        wbuf.write(&weight).unwrap();
        let xbuf = engine.alloc_host_coherent_storage((k * 4) as u64).unwrap();
        xbuf.write(bytemuck::cast_slice(&x)).unwrap();
        let out = engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        let pc = mv_pc(k, n, 1);

        let cb = engine.begin_batch().unwrap();
        let res_dead = engine.record_to(
            cb, "mul_mat_vec_f16_f32_f32_subgroup", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n as u32, 1, 1),
        );
        assert!(
            res_dead.is_err(),
            "expected 'mul_mat_vec_f16_f32_f32_subgroup' to no longer be compiled \
             (dead code, never dispatched anywhere), but record_to succeeded: {res_dead:?}"
        );

        // Its f32 counterpart (actually dispatched from vulkan_ops.py) and
        // the non-subgroup f16 base variant must still work.
        let res_f32_subgroup = engine.record_to(
            cb, "mul_mat_vec_f32_f32_f32_subgroup", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n as u32, 1, 1),
        );
        assert!(res_f32_subgroup.is_ok(), "mul_mat_vec_f32_f32_f32_subgroup should still be compiled: {res_f32_subgroup:?}");
        let res_f16 = engine.record_to(cb, "mul_mat_vec_f16_f32_f32", &[&wbuf, &xbuf, &out], bytemuck::cast_slice(&pc), (n as u32, 1, 1));
        assert!(res_f16.is_ok(), "mul_mat_vec_f16_f32_f32 should still be compiled: {res_f16:?}");
    }

    #[test]
    fn unreferenced_shader_families_are_no_longer_compiled() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        // Trivial dummy buffers/push-constants — see
        // r2_matvec_variant_is_no_longer_compiled above for why numerical
        // correctness (and matching each shader's "real" binding count)
        // is irrelevant to this test: record_to's first and only
        // observable-here check is pipeline existence, before any actual
        // GPU submission.
        let dummy = engine.alloc_host_coherent_storage(1024).unwrap();
        let pc = [0u8; 32];
        let cb = engine.begin_batch().unwrap();

        // All 37 shader names below used to be registered (compiled via
        // `spv!()` in include_all_shaders) but were never actually
        // dispatched anywhere in this codebase (Rust or Python) —
        // confirmed via `grep -rn '"<name>"' src/ vllm_vulkan/ tests/`
        // finding matches only in scripts/compile_shaders.sh (which
        // compiles every .comp source into a .spv file regardless of
        // whether Rust ever loads it) or doc-comment examples, never a
        // real `record_to`/`execute_*` call site. Removing them dropped
        // `PipelineCache::new()`'s (i.e. model-load) wall time from
        // ~707ms to ~462-464ms, consistently reproducible across
        // repeated runs — on top of #52/#53/#54's earlier matvec-focused
        // cleanup (which took it from ~1865ms to ~707ms), a combined
        // ~75% model-load startup-time reduction this session.
        for name in [
            "silu_f32", "gelu_quick_f32", "relu_f32", "exp_f32", "sigmoid_f32",
            "abs_f32", "neg_f32", "ceil_f32", "gelu_inplace_f32",
            "rms_norm_mul_rope_f32_f32", "rms_norm_mul_rope_f32_f16",
            "soft_max_f32_f16", "soft_max_large1_f32_f16", "soft_max_large2_f32_f16", "soft_max_large3_f32_f16",
            "add_f32_f32_f16", "add_f32_f16_f32", "mul_f32_f32_f16", "div_f32_f32_f16", "sub_f32_f32_f16",
            "add_rms_f32_f32_f32", "add_rms_f32_f32_f16",
            "swiglu_f32", "geglu_f32",
            "rope_norm_f32_f16", "rope_norm_f16", "rope_neox_f32_f16", "rope_neox_f16",
            "rope_multi_f32_f16", "rope_multi_f16",
            "get_rows_f32_f32", "get_rows_f16", "fill_f16", "concat_f16",
            "contig_cpy_f32_f16", "contig_cpy_f16_f32",
            "quantize_q8_1_x4",
        ] {
            let res = engine.record_to(cb, name, &[&dummy], &pc, (1, 1, 1));
            assert!(
                res.is_err(),
                "expected shader '{name}' to no longer be compiled (dead code, never \
                 dispatched anywhere), but record_to succeeded: {res:?}"
            );
        }

        // The shaders actually used in production must still work.
        for name in [
            "gelu_f32", "tanh_f32", "add_f32_f32_f32", "mul_f32_f32_f32",
            "rms_norm_f32", "rms_norm_f32_mul",
            "mul_mat_vec_f32_f32_f32", "mul_mat_vec_f16_f32_f32", "mul_mat_vec_f32_f32_f32_subgroup",
            "paged_kv_write_f16", "paged_kv_write_f32",
            "paged_attn_decode_f16", "paged_attn_decode_f16_coop", "paged_attn_decode_f16_coop_512",
            "paged_attn_decode_f32", "paged_attn_decode_f32_coop", "paged_attn_decode_f32_coop_512",
        ] {
            let res = engine.record_to(cb, name, &[&dummy], &pc, (1, 1, 1));
            assert!(res.is_ok(), "shader '{name}' should still be compiled: {res:?}");
        }
    }
}

#[cfg(test)]
mod record_to_tests {
    //! Validates `ComputeEngine::record_to`'s stack-allocated descriptor-
    //! write buffers (see its doc comment) — specifically the
    //! `buffers.len() > MAX_BINDINGS` guard, which returns a real `Err`
    //! (not a panic) since `debug_assert!` would be compiled out in
    //! release builds, and a naive fixed-size-array write beyond that
    //! bound would otherwise be the only thing standing between an
    //! over-large `buffers` slice and a crash.
    use super::*;
    use crate::pipeline::MAX_BINDINGS;

    fn make_engine() -> Option<compute::ComputeEngine> {
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: no Vulkan device available ({e})"); return None; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        Some(compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).expect("create ComputeEngine"))
    }

    #[test]
    fn record_to_rejects_more_buffers_than_max_bindings() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        // One more binding than the descriptor set layout supports.
        let too_many = MAX_BINDINGS as usize + 1;
        let bufs: Vec<compute::Buffer> = (0..too_many)
            .map(|_| engine.alloc_host_coherent_storage(4).unwrap())
            .collect();
        let buf_refs: Vec<&compute::Buffer> = bufs.iter().collect();

        let cb = engine.begin_batch().unwrap();
        let result = engine.record_to(cb, "gelu_f32", &buf_refs, &[0u8; 24], (1, 1, 1));

        assert!(
            result.is_err(),
            "record_to should reject {too_many} buffers (> MAX_BINDINGS={MAX_BINDINGS}) with an Err, not panic or silently succeed"
        );
    }

    #[test]
    fn record_to_accepts_exactly_max_bindings() {
        let _guard = gpu_test_guard();
        let Some(mut engine) = make_engine() else { return };

        let exactly_max = MAX_BINDINGS as usize;
        let bufs: Vec<compute::Buffer> = (0..exactly_max)
            .map(|_| engine.alloc_host_coherent_storage(4).unwrap())
            .collect();
        let buf_refs: Vec<&compute::Buffer> = bufs.iter().collect();

        let cb = engine.begin_batch().unwrap();
        let result = engine.record_to(cb, "gelu_f32", &buf_refs, &[0u8; 24], (1, 1, 1));

        assert!(
            result.is_ok(),
            "record_to should accept exactly MAX_BINDINGS ({MAX_BINDINGS}) buffers: {result:?}"
        );
    }
}

#[cfg(test)]
mod qkv_fusion_tests {
    //! Validates the concatenated Q+K+V weight (`self_attn.qkv_proj.weight`,
    //! built in `new()`) and its single-matvec dispatch (`use_fused_qkv` in
    //! `forward_layer_gpu_matmuls`) against three separate Q/K/V matvec
    //! dispatches against the same underlying weights. Requires a real
    //! Vulkan device; skips cleanly (not a failure) on headless CI runners
    //! with no GPU/ICD.
    use super::*;

    fn fake_random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..len).map(|_| {
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545F4914F6CDD1D);
            ((bits >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }).collect()
    }

    fn matvec_pc(k: usize, n: usize) -> Vec<u8> {
        use std::io::Write;
        let mut v = Vec::with_capacity(13 * 4);
        for x in [k as u32, k as u32, k as u32, n as u32,
                   (k * n) as u32, k as u32, n as u32,
                   0u32, 0u32, 1u32, 1u32, 1u32, 1u32] {
            v.write_all(&x.to_le_bytes()).unwrap();
        }
        v
    }

    struct Harness { engine: compute::ComputeEngine }

    fn make_harness() -> Option<Harness> {
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: no Vulkan device available ({e})"); return None; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let engine = compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).expect("create ComputeEngine");
        Some(Harness { engine })
    }

    /// Uploads f16-quantized weight/input once, then supports dispatching
    /// against them repeatedly (mirrors production: weights are uploaded
    /// once at model-load time, dispatched every decode step afterwards).
    struct Fixture {
        q_buf: compute::Buffer,
        k_buf: compute::Buffer,
        v_buf: compute::Buffer,
        qkv_buf: compute::Buffer,
        inp: compute::Buffer,
        q_dim: usize,
        kv_dim: usize,
        k: usize,
    }

    fn build_fixture(h: &mut Harness, k: usize, q_dim: usize, kv_dim: usize) -> Fixture {
        let qw = fake_random(q_dim * k, 1);
        let kw = fake_random(kv_dim * k, 2);
        let vw = fake_random(kv_dim * k, 3);
        let x = fake_random(k, 4);

        let upload = |eng: &mut compute::ComputeEngine, data: &[f32]| {
            let bytes = f32_to_f16_bytes(data);
            let buf = eng.alloc_host_coherent_storage(bytes.len() as u64).unwrap();
            buf.write(&bytes).unwrap();
            buf
        };
        let q_buf = upload(&mut h.engine, &qw);
        let k_buf = upload(&mut h.engine, &kw);
        let v_buf = upload(&mut h.engine, &vw);

        // Exactly what new()'s weight-upload loop does: concatenate the
        // raw f32 rows, then quantize the concatenation as one f16 buffer.
        let mut combined = Vec::with_capacity(qw.len() + kw.len() + vw.len());
        combined.extend_from_slice(&qw);
        combined.extend_from_slice(&kw);
        combined.extend_from_slice(&vw);
        let qkv_buf = upload(&mut h.engine, &combined);

        let xb: &[u8] = bytemuck::cast_slice(&x);
        let inp = h.engine.alloc_host_coherent_storage(xb.len() as u64).unwrap();
        inp.write(xb).unwrap();

        Fixture { q_buf, k_buf, v_buf, qkv_buf, inp, q_dim, kv_dim, k }
    }

    fn dispatch_separate(h: &mut Harness, f: &Fixture) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_out = h.engine.alloc_host_coherent_storage((f.q_dim * 4) as u64).unwrap();
        let k_out = h.engine.alloc_host_coherent_storage((f.kv_dim * 4) as u64).unwrap();
        let v_out = h.engine.alloc_host_coherent_storage((f.kv_dim * 4) as u64).unwrap();
        let cb = h.engine.begin_batch().unwrap();
        h.engine.record_to(cb, "mul_mat_vec_f16_f32_f32_r4", &[&f.q_buf, &f.inp, &q_out], &matvec_pc(f.k, f.q_dim), (wg_r4(f.q_dim), 1, 1)).unwrap();
        h.engine.record_to(cb, "mul_mat_vec_f16_f32_f32_r4", &[&f.k_buf, &f.inp, &k_out], &matvec_pc(f.k, f.kv_dim), (wg_r4(f.kv_dim), 1, 1)).unwrap();
        h.engine.record_to(cb, "mul_mat_vec_f16_f32_f32_r4", &[&f.v_buf, &f.inp, &v_out], &matvec_pc(f.k, f.kv_dim), (wg_r4(f.kv_dim), 1, 1)).unwrap();
        h.engine.submit_batch(cb).unwrap();
        (read_f32_buf(&q_out, f.q_dim), read_f32_buf(&k_out, f.kv_dim), read_f32_buf(&v_out, f.kv_dim))
    }

    fn dispatch_fused(h: &mut Harness, f: &Fixture) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = f.q_dim + 2 * f.kv_dim;
        let out = h.engine.alloc_host_coherent_storage((n * 4) as u64).unwrap();
        let cb = h.engine.begin_batch().unwrap();
        h.engine.record_to(cb, "mul_mat_vec_f16_f32_f32_r4", &[&f.qkv_buf, &f.inp, &out], &matvec_pc(f.k, n), (wg_r4(n), 1, 1)).unwrap();
        h.engine.submit_batch(cb).unwrap();
        (
            read_f32_buf_at(&out, 0, f.q_dim),
            read_f32_buf_at(&out, f.q_dim, f.kv_dim),
            read_f32_buf_at(&out, f.q_dim + f.kv_dim, f.kv_dim),
        )
    }

    #[test]
    fn fused_qkv_matches_separate_dispatch() {
        let _guard = gpu_test_guard();
        let Some(mut h) = make_harness() else { return };
        // Both Gemma4-E2B QKV shapes (sliding-window and full-attention layers).
        for (label, q_dim, kv_dim) in [("sliding", 2048, 256), ("full-attn", 4096, 512)] {
            let f = build_fixture(&mut h, 1536, q_dim, kv_dim);
            let (q_sep, k_sep, v_sep) = dispatch_separate(&mut h, &f);
            let (q_fused, k_fused, v_fused) = dispatch_fused(&mut h, &f);
            for (i, (&a, &b)) in q_sep.iter().zip(q_fused.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "{label}: Q mismatch at {i}: {a} vs {b}");
            }
            for (i, (&a, &b)) in k_sep.iter().zip(k_fused.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "{label}: K mismatch at {i}: {a} vs {b}");
            }
            for (i, (&a, &b)) in v_sep.iter().zip(v_fused.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "{label}: V mismatch at {i}: {a} vs {b}");
            }
            println!("{label:<10} fused QKV matches 3 separate dispatches exactly (q_dim={q_dim} kv_dim={kv_dim})");
        }
    }

    #[test]
    fn fused_qkv_is_faster_than_separate_dispatch() {
        let _guard = gpu_test_guard();
        let Some(mut h) = make_harness() else { return };
        let f = build_fixture(&mut h, 1536, 2048, 256); // sliding-window shape

        for _ in 0..5 { dispatch_separate(&mut h, &f); dispatch_fused(&mut h, &f); }

        let iters = 1000;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { dispatch_separate(&mut h, &f); }
        let sep_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters { dispatch_fused(&mut h, &f); }
        let fused_us = t0.elapsed().as_micros() as f64 / iters as f64;

        println!(
            "QKV: 3 separate _r4 dispatches {sep_us:.1}us/call   1 fused _r4 dispatch {fused_us:.1}us/call   speedup {:.2}x",
            sep_us / fused_us
        );
        assert!(
            fused_us < sep_us,
            "fused QKV ({fused_us:.1}us) was not faster than 3 separate dispatches ({sep_us:.1}us)"
        );
    }
}

#[cfg(test)]
mod softcap_tests {
    //! Validates the GPU-based final logit softcap (`forward_gpu`'s
    //! broadcast-mul(1/cap) -> tanh -> broadcast-mul(cap) chain, appended
    //! to the LM head's existing matvec submit) against the single-
    //! threaded CPU loop it replaces (`(x/cap).tanh()*cap` over the whole
    //! vocab — 262144 elements for Gemma4-E2B, ~1.1ms/call in isolation).
    //! Requires a real Vulkan device; skips cleanly (not a failure) on
    //! headless CI runners with no GPU/ICD.
    use super::*;

    #[test]
    fn gpu_softcap_vs_cpu_softcap() {
        let _guard = gpu_test_guard();
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: {e}"); return; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let mut engine = compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).unwrap();

        let vocab = 262144usize;
        let cap = 30.0f32;
        let logits_init: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.0001).sin() * 40.0).collect();

        // CPU reference (today's implementation).
        let cpu_softcap = |logits: &[f32]| -> Vec<f32> {
            logits.iter().map(|&l| (l / cap).tanh() * cap).collect()
        };
        let cpu_result = cpu_softcap(&logits_init);

        // GPU: broadcast-mul(1/cap) -> tanh -> broadcast-mul(cap), all in one submit.
        let logits_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        logits_buf.write(bytemuck::cast_slice(&logits_init)).unwrap();
        let scaled_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        let tanh_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        let final_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        let inv_cap_buf = engine.alloc_host_coherent_storage(4).unwrap();
        inv_cap_buf.write(bytemuck::cast_slice(&[1.0f32 / cap])).unwrap();
        let cap_buf = engine.alloc_host_coherent_storage(4).unwrap();
        cap_buf.write(bytemuck::cast_slice(&[cap])).unwrap();

        let gpu_softcap = |eng: &mut compute::ComputeEngine| -> Vec<f32> {
            let cb = eng.begin_batch().unwrap();
            eng.record_to(cb, "mul_f32_f32_f32", &[&logits_buf, &inv_cap_buf, &scaled_buf], bytemuck::cast_slice(&binary_broadcast_pc(vocab)), (wg256(vocab), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "tanh_f32", &[&scaled_buf, &tanh_buf], bytemuck::cast_slice(&unary_head_pc(vocab)), (wg128(vocab), 1, 1)).unwrap();
            eng.record_barrier_to(cb);
            eng.record_to(cb, "mul_f32_f32_f32", &[&tanh_buf, &cap_buf, &final_buf], bytemuck::cast_slice(&binary_broadcast_pc(vocab)), (wg256(vocab), 1, 1)).unwrap();
            eng.submit_batch(cb).unwrap();
            read_f32_buf(&final_buf, vocab)
        };

        let gpu_result = gpu_softcap(&mut engine);
        let mut max_err = 0.0f32;
        for (&a, &b) in cpu_result.iter().zip(gpu_result.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        println!("GPU vs CPU softcap max abs err: {max_err:.6}");
        assert!(max_err < 1e-4, "GPU softcap diverged from CPU reference: {max_err}");

        // Timing.
        for _ in 0..3 {
            cpu_softcap(&logits_init);
            gpu_softcap(&mut engine);
        }
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { std::hint::black_box(cpu_softcap(&logits_init)); }
        let cpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters { std::hint::black_box(gpu_softcap(&mut engine)); }
        let gpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        println!(
            "softcap over {vocab} elements: CPU {cpu_us:.1}us/call   GPU (3 dispatches, 1 submit) {gpu_us:.1}us/call   speedup {:.2}x",
            cpu_us / gpu_us
        );
    }
}

#[cfg(test)]
mod ple_proj_tests {
    //! Validates `gpu_matmul_or_cpu` — the method `forward_gpu`'s PLE
    //! (per-layer embedding) preprocessing now uses for its
    //! "per_layer_model_projection" matmul ([1,H] x [total_ple,H]^T,
    //! H=1536, total_ple=8960 for Gemma4-E2B) — against the
    //! `model::cpu_matmul` (matrixmultiply::sgemm) path it replaces there.
    //! Unlike every per-layer projection (q/k/v/o_proj/gate/up/down), this
    //! one previously always ran on the CPU regardless of GPU
    //! availability; measured in isolation that CPU matmul takes ~6.9ms
    //! (confirmed against a naive dot-product loop too, which is *slower*
    //! — this isn't a GEMM-packing-overhead artifact, just real FLOPs).
    //! Requires a real Vulkan device; skips cleanly (not a failure) on
    //! headless CI runners with no GPU/ICD.
    use super::*;
    use super::matvec_fusion_tests::build_test_model;

    fn fake_random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        (0..len).map(|_| {
            state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545F4914F6CDD1D);
            ((bits >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }).collect()
    }

    /// Whole-vector L2 relative error — see other test modules' identical
    /// helper (e.g. `matvec_fusion_tests::l2_rel_err`) for why this is used
    /// instead of a per-element relative error.
    fn l2_rel_err(a: &[f32], b: &[f32]) -> f32 {
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            diff_sq += ((x - y) as f64).powi(2);
            ref_sq += (x as f64).powi(2);
        }
        (diff_sq / ref_sq.max(1e-12)).sqrt() as f32
    }

    #[test]
    fn gpu_matmul_or_cpu_matches_cpu_matmul_for_ple_proj() {
        let _guard = gpu_test_guard();
        let Some(mut model) = build_test_model(true) else { return };

        let cfg = &model.inner.config;
        let h = cfg.hidden_size;
        let total_ple = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
        let hidden = fake_random(h, 100);

        let cpu_result = model::cpu_matmul(
            &hidden,
            model.inner.weights.f32_slice("model.per_layer_model_projection.weight"),
            1, h, total_ple,
        );

        assert!(model.gpu_weights.contains_key("model.per_layer_model_projection.weight"));
        let pc_vals: [u32; 13] = [
            h as u32, h as u32, h as u32, total_ple as u32,
            (h * total_ple) as u32, h as u32, total_ple as u32,
            0u32, 0u32, 1u32, 1u32, 1u32, 1u32,
        ];
        let pc: &[u8] = bytemuck::cast_slice(&pc_vals);
        let gpu_result = model.gpu_matmul_or_cpu(
            "model.per_layer_model_projection.weight", &hidden, 1, h, total_ple, pc,
        );

        let err = l2_rel_err(&cpu_result, &gpu_result);
        println!("gpu_matmul_or_cpu (PLE-proj shape) vs cpu_matmul: l2_rel_err={err:.6}");
        assert!(err < 0.01, "gpu_matmul_or_cpu diverged from cpu_matmul reference: {err}");
    }

    #[test]
    fn gpu_matmul_or_cpu_is_faster_than_cpu_matmul_for_ple_proj() {
        let _guard = gpu_test_guard();
        let Some(mut model) = build_test_model(true) else { return };

        let cfg = model.inner.config.clone();
        let h = cfg.hidden_size;
        let total_ple = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input;
        let hidden = fake_random(h, 200);
        let weight = model.inner.weights.f32_slice("model.per_layer_model_projection.weight").to_vec();
        let pc_vals: [u32; 13] = [
            h as u32, h as u32, h as u32, total_ple as u32,
            (h * total_ple) as u32, h as u32, total_ple as u32,
            0u32, 0u32, 1u32, 1u32, 1u32, 1u32,
        ];
        let pc: &[u8] = bytemuck::cast_slice(&pc_vals);

        for _ in 0..3 {
            model::cpu_matmul(&hidden, &weight, 1, h, total_ple);
            model.gpu_matmul_or_cpu("model.per_layer_model_projection.weight", &hidden, 1, h, total_ple, pc);
        }

        let iters = 30;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { std::hint::black_box(model::cpu_matmul(&hidden, &weight, 1, h, total_ple)); }
        let cpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(model.gpu_matmul_or_cpu("model.per_layer_model_projection.weight", &hidden, 1, h, total_ple, pc));
        }
        let gpu_us = t0.elapsed().as_micros() as f64 / iters as f64;

        println!(
            "PLE-proj matmul [1,{h}] x [{total_ple},{h}]^T via gpu_matmul_or_cpu: CPU(sgemm) {cpu_us:.1}us/call   GPU {gpu_us:.1}us/call   speedup {:.2}x",
            cpu_us / gpu_us
        );
        assert!(
            gpu_us < cpu_us,
            "gpu_matmul_or_cpu ({gpu_us:.1}us) was not faster than cpu_matmul ({cpu_us:.1}us)"
        );
    }
}

#[cfg(test)]
mod final_norm_lm_head_tests {
    //! Validates `forward_gpu`'s final `model.norm` RMSNorm — now dispatched
    //! as `rms_norm_f32_mul` at the start of the LM head's existing
    //! matvec+softcap submit instead of as a separate CPU `cpu_rms_norm`
    //! call beforehand — against the CPU reference chain it replaces
    //! (`cpu_rms_norm` -> `cpu_matmul` -> softcap). Same GPU-fused-norm
    //! pattern `gpu_input_layernorm_matches_cpu_reference` above already
    //! validates for the QKV submit's `input_layernorm`, applied to the
    //! other end of the decoder stack.
    //!
    //! Uses a reduced vocab size (4096, vs. Gemma4-E2B's real 262144) so
    //! the test allocates a `[4096, 1536]` f32 weight (~25MB) instead of a
    //! `[262144, 1536]` one (~1.6GB) — `model.embed_tokens.weight` stays
    //! f32 (LM head precision, see `is_matvec_weight`'s doc comment), so
    //! unlike PLE-proj's f16 weight this isn't a case where a smaller test
    //! risks missing a size-dependent bug in the matvec kernel itself:
    //! `mul_mat_vec_f32_f32_f32_r4` (and its f16-weight sibling) is already
    //! validated bit-for-bit against the base (non-r4) kernel at multiple
    //! shapes by `matvec_r4_tests` above. What's new and actually under
    //! test here is the *ordering* — that `rms_norm_f32_mul`'s output
    //! correctly feeds the matvec dispatch that follows it in the same
    //! command buffer — which a reduced N exercises identically to the
    //! real one (both are well above the r4 kernel's 4-rows-per-workgroup
    //! tiling granularity).
    //!
    //! Requires a real Vulkan device; skips cleanly (not a failure) on
    //! headless CI runners with no GPU/ICD.
    use super::*;

    fn fake_random(len: usize, seed: u64) -> Vec<f32> {
        (0..len).map(|i| {
            let x = (i as u64).wrapping_mul(2654435761).wrapping_add(seed);
            ((x % 20000) as f32 / 10000.0) - 1.0
        }).collect()
    }

    fn l2_rel_err(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>().sqrt();
        let den: f32 = a.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
        if den > 0.0 { num / den } else { num }
    }

    /// CPU reference: `cpu_rms_norm` -> `cpu_matmul` -> softcap — exactly
    /// what forward_gpu's `else` (no-GPU) branch still does.
    fn cpu_reference(hidden: &[f32], norm_w: &[f32], lm_w: &[f32], h: usize, vocab: usize, eps: f32, cap: f32) -> Vec<f32> {
        let normed = model::cpu_rms_norm(hidden, norm_w, eps);
        let mut raw = model::cpu_matmul(&normed, lm_w, 1, h, vocab);
        raw.iter_mut().for_each(|l| *l = (*l / cap).tanh() * cap);
        raw
    }

    #[test]
    fn gpu_fused_final_norm_matches_cpu_reference() {
        let _guard = gpu_test_guard();
        let dev = match device::ComputeDevice::create(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip: {e}"); return; }
        };
        let shader_spvs = include_all_shaders();
        let refs: std::collections::HashMap<&str, &[u8]> = shader_spvs.iter()
            .map(|(k, v)| (k.as_str(), v.as_slice())).collect();
        let mut engine = compute::ComputeEngine::new(
            dev.instance.clone(), dev.physical_device, dev.device.clone(),
            dev.compute_queue, dev.compute_queue_family, &refs,
        ).unwrap();

        let h = 1536usize;
        let vocab = 4096usize; // see module doc comment for why this is reduced
        let eps = 1e-6f32;
        let cap = 30.0f32;

        let hidden = fake_random(h, 400);
        let norm_w = fake_random(h, 401);
        let lm_w = fake_random(vocab * h, 402);

        let cpu_result = cpu_reference(&hidden, &norm_w, &lm_w, h, vocab, eps, cap);

        // GPU: rms_norm_f32_mul -> mul_mat_vec_f32_f32_f32_r4 -> softcap
        // chain, exactly forward_gpu's use_gpu_lm_head path, in one submit.
        let raw_hidden_buf = engine.alloc_host_coherent_storage((h * 4) as u64).unwrap();
        raw_hidden_buf.write(bytemuck::cast_slice(&hidden)).unwrap();
        let norm_w_buf = engine.alloc_host_coherent_storage((h * 4) as u64).unwrap();
        norm_w_buf.write(bytemuck::cast_slice(&norm_w)).unwrap();
        let lm_w_buf = engine.alloc_host_coherent_storage((vocab * h * 4) as u64).unwrap();
        lm_w_buf.write(bytemuck::cast_slice(&lm_w)).unwrap();
        let normed_buf = engine.alloc_host_coherent_storage((h * 4) as u64).unwrap();
        let raw_logit_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        let scaled_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        let tanh_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        let final_buf = engine.alloc_host_coherent_storage((vocab * 4) as u64).unwrap();
        let inv_cap_buf = engine.alloc_host_coherent_storage(4).unwrap();
        inv_cap_buf.write(bytemuck::cast_slice(&[1.0f32 / cap])).unwrap();
        let cap_buf = engine.alloc_host_coherent_storage(4).unwrap();
        cap_buf.write(bytemuck::cast_slice(&[cap])).unwrap();

        let pc_vals: [u32; 13] = [
            h as u32, h as u32, h as u32, vocab as u32,
            (h * vocab) as u32, h as u32, vocab as u32,
            0u32, 0u32, 1u32, 1u32, 1u32, 1u32,
        ];
        let pc: &[u8] = bytemuck::cast_slice(&pc_vals);

        let cb = engine.begin_batch().unwrap();
        engine.record_to(cb, "rms_norm_f32_mul", &[&raw_hidden_buf, &norm_w_buf, &normed_buf], bytemuck::cast_slice(&rms_norm_mul_pc(h, eps)), (1, 1, 1)).unwrap();
        engine.record_barrier_to(cb);
        engine.record_to(cb, "mul_mat_vec_f32_f32_f32_r4", &[&lm_w_buf, &normed_buf, &raw_logit_buf], pc, (wg_r4(vocab), 1, 1)).unwrap();
        engine.record_barrier_to(cb);
        engine.record_to(cb, "mul_f32_f32_f32", &[&raw_logit_buf, &inv_cap_buf, &scaled_buf], bytemuck::cast_slice(&binary_broadcast_pc(vocab)), (wg256(vocab), 1, 1)).unwrap();
        engine.record_barrier_to(cb);
        engine.record_to(cb, "tanh_f32", &[&scaled_buf, &tanh_buf], bytemuck::cast_slice(&unary_head_pc(vocab)), (wg128(vocab), 1, 1)).unwrap();
        engine.record_barrier_to(cb);
        engine.record_to(cb, "mul_f32_f32_f32", &[&tanh_buf, &cap_buf, &final_buf], bytemuck::cast_slice(&binary_broadcast_pc(vocab)), (wg256(vocab), 1, 1)).unwrap();
        engine.submit_batch(cb).unwrap();

        let gpu_result = read_f32_buf(&final_buf, vocab);

        let err = l2_rel_err(&cpu_result, &gpu_result);
        println!("GPU fused final-norm+LM-head+softcap vs CPU reference: l2_rel_err={err:.6}");
        assert!(err < 1e-4, "GPU fused final norm diverged from CPU reference: {err}");
    }
}
