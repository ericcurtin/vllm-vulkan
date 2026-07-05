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

    // ── Elementwise unary (f32 only — new llama.cpp has no f16 variants) ─
    spv!("silu_f32");
    spv!("gelu_f32");
    spv!("gelu_quick_f32");
    spv!("relu_f32");
    spv!("tanh_f32");
    spv!("exp_f32");
    spv!("sigmoid_f32");
    spv!("abs_f32");
    spv!("neg_f32");
    spv!("ceil_f32");

    // ── Normalization ───────────────────────────────────────────────────
    // rms_norm: fused with optional RoPE (ADD_RMS=0 means plain rms_norm)
    spv!("rms_norm_mul_rope_f32_f32");
    spv!("rms_norm_mul_rope_f32_f16");

    // ── Softmax ─────────────────────────────────────────────────────────
    spv!("soft_max_f32_f16");
    spv!("soft_max_large1_f32_f16");
    spv!("soft_max_large2_f32_f16");
    spv!("soft_max_large3_f32_f16");

    // ── Binary elementwise ──────────────────────────────────────────────
    spv!("add_f32_f32_f16");
    spv!("add_f32_f16_f32");
    spv!("mul_f32_f32_f16");
    spv!("div_f32_f32_f16");
    spv!("sub_f32_f32_f16");
    spv!("add_rms_f32_f32_f32");
    spv!("add_rms_f32_f32_f16");

    // ── GLU activations ─────────────────────────────────────────────────
    spv!("swiglu_f32");
    spv!("geglu_f32");

    // ── RoPE ────────────────────────────────────────────────────────────
    spv!("rope_norm_f32_f16");
    spv!("rope_norm_f16");
    spv!("rope_neox_f32_f16");
    spv!("rope_neox_f16");
    spv!("rope_multi_f32_f16");
    spv!("rope_multi_f16");

    // ── Copy / reshape ──────────────────────────────────────────────────
    spv!("get_rows_f32_f32");
    spv!("get_rows_f16");
    spv!("fill_f16");
    spv!("concat_f16");
    spv!("contig_cpy_f32_f16");
    spv!("contig_cpy_f16_f32");

    // ── Matmul (decode: matrix-vector) ──────────────────────────────────
    // f32/f16 weights → f32 output
    spv!("mul_mat_vec_f32_f32_f32_subgroup");
    spv!("mul_mat_vec_f16_f32_f32");
    spv!("mul_mat_vec_f16_f32_f32_subgroup");
    // Quantized weight × f32 activations → f32 output
    spv!("mul_mat_vec_q4_0_f32_f32");
    spv!("mul_mat_vec_q4_0_f32_f32_subgroup");
    spv!("mul_mat_vec_q4_1_f32_f32");
    spv!("mul_mat_vec_q5_0_f32_f32");
    spv!("mul_mat_vec_q5_1_f32_f32");
    spv!("mul_mat_vec_q8_0_f32_f32");
    spv!("mul_mat_vec_q8_0_f32_f32_subgroup");
    // Quant K-type: activations f16 → f32 (new naming convention)
    spv!("mul_mat_vec_q2_k_f16_f32");
    spv!("mul_mat_vec_q3_k_f16_f32");
    spv!("mul_mat_vec_q4_k_f32_f32_subgroup");
    spv!("mul_mat_vec_q5_k_f16_f32");
    spv!("mul_mat_vec_q6_k_f32_f32_subgroup");
    spv!("mul_mat_vec_iq4_nl_f32_f32");

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
    spv!("quantize_q8_1_x4");

    // ── Manually compiled extras (needed for GPU inference) ──────────────
    // These were compiled directly with glslangValidator (not via vulkan-shaders-gen).
    spv!("rms_norm_f32");
    spv!("mul_f32_f32_f32");   // elementwise multiply f32×f32→f32
    spv!("gelu_inplace_f32");  // GELU activation f32→f32
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
    spv!("paged_attn_decode_f32");
    spv!("paged_attn_decode_f32_coop");

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
const ACT_COUNT:   usize = 17;

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
                        log::info!("Uploaded {:.1}GB of weights to GPU (projection weights as f16)",
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

        // ── PLE preprocessing (CPU) ──────────────────────────────────────────
        let ple_embed_w = self.inner.weights.f32_slice("model.embed_tokens_per_layer.weight");
        let ple_embeds_flat: Vec<f32> = ple_embed_w[token_id as usize * total_ple..
                                                       (token_id as usize + 1) * total_ple]
            .iter().map(|&v| v * cfg.ple_scale).collect();

        let proj_w = self.inner.weights.f32_slice("model.per_layer_model_projection.weight");
        let ple_proj = model::cpu_matmul(&hidden, proj_w, 1, h, total_ple);
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
        let norm_w = self.inner.weights.f32_slice("model.norm.weight");
        let normed = model::cpu_rms_norm(&hidden, norm_w, eps);

        // LM head: [1, H] @ [vocab, H]^T — the biggest matmul, must be on GPU.
        let vocab = cfg.vocab_size;
        let cap = cfg.final_logit_softcapping;
        let mut logits = if let (Some(eng), Some(lm_w_ptr)) = (
            self.engine.as_mut(),
            self.gpu_weights.get("model.embed_tokens.weight")
                .map(|b| b as *const compute::Buffer)
        ) {
            let normed_bytes = f32_slice_to_bytes(&normed);
            let logit_size = (vocab * 4) as u64;

            // Use persistent buffers for LM head too
            let inp_p  = self.act_ptr_mut(ACT_QKV_IN);  // reuse - we're done with layer ops
            let out_p_key = logit_size + 99;  // unique key for logit output
            // Allocate logit output buffer lazily
            let logit_p: *const compute::Buffer = {
                if !self.act_bufs.iter().any(|b| b.size == logit_size) {
                    if let Ok(buf) = self.engine.as_mut().unwrap().alloc_host_coherent_storage(logit_size) {
                        self.act_bufs.push(buf);
                    }
                }
                // Find the logit buffer
                self.act_bufs.iter().find(|b| b.size == logit_size)
                    .map(|b| b as *const compute::Buffer)
                    .unwrap_or(std::ptr::null())
            };

            if logit_p.is_null() {
                // Fallback to CPU
                let lm_w = self.inner.weights.f32_slice("model.embed_tokens.weight");
                model::cpu_matmul(&normed, lm_w, 1, h, vocab)
            } else {
                unsafe { (*inp_p).write(&normed_bytes).unwrap(); }

                let eng = self.engine.as_mut().unwrap();
                let pc = {
                    use std::io::Write;
                    let mut v = Vec::with_capacity(13 * 4);
                    for x in [h as u32, h as u32, h as u32, vocab as u32,
                               (h * vocab) as u32, h as u32, vocab as u32,
                               0u32, 0u32, 1u32, 1u32, 1u32, 1u32] {
                        v.write_all(&x.to_le_bytes()).unwrap();
                    }
                    v
                };
                let cb = eng.begin_batch().unwrap();
                let inp_ref = inp_p as *const compute::Buffer;
                unsafe {
                    eng.record_to(cb, "mul_mat_vec_f32_f32_f32",
                        &[&*lm_w_ptr, &*inp_ref, &*logit_p],
                        &pc, (vocab as u32, 1, 1)).unwrap();
                }
                eng.submit_batch(cb).unwrap();
                read_f32_buf(unsafe { &*logit_p }, vocab)
            }
        } else {
            let lm_w = self.inner.weights.f32_slice("model.embed_tokens.weight");
            model::cpu_matmul(&normed, lm_w, 1, h, vocab)
        };

        logits.iter_mut().for_each(|l| *l = (*l / cap).tanh() * cap);
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
        let mv_pc = |k: usize, n: usize| -> Vec<u8> {
            use std::io::Write;
            let mut v = Vec::with_capacity(13 * 4);
            for x in [k as u32, k as u32, k as u32, n as u32,
                       (k * n) as u32, k as u32, n as u32,
                       0u32, 0u32, 1u32, t as u32, t as u32, 1u32] {
                v.write_all(&x.to_le_bytes()).unwrap();
            }
            v
        };

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
        let residual = hidden.to_vec();
        let _t_layer = std::time::Instant::now();

        // CPU: input_layernorm
        let x = model::cpu_rms_norm(hidden, &inln_w, eps);

        let xb = f32_slice_to_bytes(&x);
        let shader = "mul_mat_vec_f16_f32_f32";

        // Init persistent activation buffers on first call.
        let use_gpu = self.engine.is_some()
            && self.gpu_weights.contains_key(&ln("self_attn.q_proj.weight"))
            && self.init_act_bufs();

        // ── GPU BATCH: ALL 7 MATMULS IN ONE vkQueueSubmit ───────────────────
        // We batch QKV + o_proj + gate + up + down into a single command buffer.
        // The fence wait happens ONCE per layer instead of 4 times.
        // Between QKV submit and down_proj, CPU runs: Q/K/V norms, RoPE, SDPA.
        // We split into 2 submits at the attention boundary:
        //   Submit 1: Q + K + V  (before attention)
        //   Submit 2: o_proj + gate + up + down  (after attention, combined)

        let (q_vec, k_vec, v_vec) = if use_gpu {
            // Write input to persistent buffer.
            unsafe { (*self.act_ptr_mut(ACT_QKV_IN)).write(&xb).unwrap(); }

            let inp = self.act_ptr(ACT_QKV_IN);
            let q_p = self.act_ptr(ACT_Q_OUT);
            let k_p = self.act_ptr(ACT_K_OUT);
            let v_p = self.act_ptr(ACT_V_OUT);

            let q_w = &self.gpu_weights[&ln("self_attn.q_proj.weight")] as *const compute::Buffer;
            let k_w = &self.gpu_weights[&ln("self_attn.k_proj.weight")] as *const compute::Buffer;
            let v_w = &self.gpu_weights[&ln("self_attn.v_proj.weight")] as *const compute::Buffer;

            // SUBMIT 1: Q, K, V in one command buffer (no barriers needed — independent)
            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, shader, &[&*q_w, &*inp, &*q_p], &mv_pc(h, q_dim), (q_dim as u32, t as u32, 1)).unwrap();
                eng.record_to(cb, shader, &[&*k_w, &*inp, &*k_p], &mv_pc(h, kv_dim), (kv_dim as u32, t as u32, 1)).unwrap();
                eng.record_to(cb, shader, &[&*v_w, &*inp, &*v_p], &mv_pc(h, kv_dim), (kv_dim as u32, t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();  // Fence wait 1: QKV
            if layer_idx == 0 { log::debug!("L{layer_idx} QKV submit: {}µs", _t_layer.elapsed().as_micros()); }

            let q_v = read_f32_buf(unsafe { &*q_p }, t * q_dim);
            let k_v = read_f32_buf(unsafe { &*k_p }, t * kv_dim);
            let v_v = read_f32_buf(unsafe { &*v_p }, t * kv_dim);
            (q_v, k_v, v_v)
        } else {
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

        // CPU: Q-norm, K-norm, V-norm (using pre-extracted weights)
        for hi in 0..num_q {
            let s = &mut q[hi * head_dim..(hi + 1) * head_dim];
            let n = model::cpu_rms_norm(s, &q_norm_w, eps);
            s.copy_from_slice(&n);
        }
        if !is_kv_shared {
            let k_norm = k_norm_w.as_ref().unwrap();
            for hi in 0..num_kv {
                let s = &mut k_final[hi * head_dim..(hi + 1) * head_dim];
                let n = model::cpu_rms_norm(s, k_norm, eps);
                s.copy_from_slice(&n);
            }
            for hi in 0..num_kv {
                let s = &mut v_final[hi * head_dim..(hi + 1) * head_dim];
                let n = model::cpu_rms_norm_no_weight(s, head_dim, eps);
                s.copy_from_slice(&n);
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

        // GPU: o_proj — use persistent buffers (no alloc/free overhead)
        let o_proj = if use_gpu && self.gpu_weights.contains_key(&ln("self_attn.o_proj.weight")) {
            let attnb = f32_slice_to_bytes(&attn_out);
            unsafe { (*self.act_ptr_mut(ACT_O_IN)).write(&attnb).unwrap(); }
            let oi   = self.act_ptr(ACT_O_IN);
            let oo   = self.act_ptr(ACT_O_OUT);
            let ow   = &self.gpu_weights[&ln("self_attn.o_proj.weight")] as *const compute::Buffer;
            let eng  = self.engine.as_mut().unwrap();
            let cb   = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, shader, &[&*ow, &*oi, &*oo], &mv_pc(q_dim, h), (h as u32, t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();
            if layer_idx == 0 { log::debug!("L{layer_idx} o_proj submit: {}µs total since layer start", _t_layer.elapsed().as_micros()); }
            read_f32_buf(unsafe { &*oo }, t * h)
        } else {
            let ow = self.inner.weights.f32_slice(&ln("self_attn.o_proj.weight"));
            model::cpu_matmul(&attn_out, ow, 1, q_dim, h)
        };

        // CPU: post_attn_norm + residual (using pre-extracted weight)
        let pa_normed = model::cpu_rms_norm(&o_proj, &pa_w, eps);
        let hidden2: Vec<f32> = residual.iter().zip(pa_normed.iter())
            .map(|(&r, &a)| r + a).collect();
        let residual2 = hidden2.clone();

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
            let ffb = f32_slice_to_bytes(&ff_in);
            unsafe { (*self.act_ptr_mut(ACT_FFIN)).write(&ffb).unwrap(); }

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
            // gelu_inplace_f32: local_size_x=512, mul_f32_f32_f32: local_size_x=256
            let gelu_wg = ((ffn_inter + 511) / 512) as u32;
            let mul_wg  = ((ffn_inter + 255) / 256) as u32;
            let ew_wg = gelu_wg; // used below for gelu dispatch
            let gelu_pc = {
                use std::io::Write;
                let mut v = Vec::with_capacity(6 * 4);
                let kx = ffn_inter as u32;
                v.write_all(&kx.to_le_bytes()).unwrap();       // KX = num elements
                v.write_all(&1u32.to_le_bytes()).unwrap();     // KY = 1
                for _ in 0..4 { v.write_all(&0u32.to_le_bytes()).unwrap(); } // param1-4
                v
            };
            // mul.comp generic_binary_head push constants for elementwise [ffn_inter] × [ffn_inter] → [ffn_inter]
            // Format: ne(uint), ne00-ne03(4 uint), nb00-nb03(4 uint), [same for src1], [same for dst], misalign(uint), param1(f32), param2(f32), param3(i32)
            // nb values are in ELEMENTS (ggml convention: nb00=1, nb01=n for a [n] flat tensor)
            let mul_pc = {
                use std::io::Write;
                let n = ffn_inter as u32;
                let mut v = Vec::with_capacity(29 * 4);
                // ne, ne00-ne03, nb00-nb03 (src0)
                for &x in &[n, n,1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                // ne10-ne13, nb10-nb13 (src1)
                for &x in &[n, 1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                // ne20-ne23, nb20-nb23 (dst)
                for &x in &[n, 1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                // misalign, param1, param2, param3
                for &x in &[0u32, 0u32, 0u32, 0u32] { v.write_all(&x.to_le_bytes()).unwrap(); }
                v
            };

            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                // Step 1: gate and up matmuls are independent (same input ffi, different outputs)
                eng.record_to(cb, shader, &[&*gw, &*ffi, &*gp], &mv_pc(h, ffn_inter), (ffn_inter as u32, t as u32, 1)).unwrap();
                eng.record_to(cb, shader, &[&*uw, &*ffi, &*up_p], &mv_pc(h, ffn_inter), (ffn_inter as u32, t as u32, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 2: gelu(gate) → gelu_p  (gelu_f32: binding0=src, binding1=dst)
                eng.record_to(cb, "gelu_f32", &[&*gp, &*gelu_p], &gelu_pc, (ew_wg, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 3: mid = gelu(gate) * up
                eng.record_to(cb, "mul_f32_f32_f32", &[&*gelu_p, &*up_p, &*mid_p], &mul_pc, (mul_wg, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                // Step 4: ff_out = down_proj(mid)
                eng.record_to(cb, shader, &[&*dw, &*mid_p, &*down_p], &mv_pc(ffn_inter, h), (h as u32, t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();  // ONE fence wait for all FFN ops
            if layer_idx == 0 { log::debug!("L{layer_idx} FFN submit: {}µs total since layer start", _t_layer.elapsed().as_micros()); }

            read_f32_buf(unsafe { &*down_p }, t * h)
        } else {
            // CPU fallback
            let gate_w = self.inner.weights.f32_slice(&ln("mlp.gate_proj.weight")).to_vec();
            let up_w   = self.inner.weights.f32_slice(&ln("mlp.up_proj.weight")).to_vec();
            let gate = model::cpu_matmul(&ff_in, &gate_w, 1, h, ffn_inter);
            let up   = model::cpu_matmul(&ff_in, &up_w,   1, h, ffn_inter);
            let gate_act = model::cpu_gelu(&gate);
            let mid: Vec<f32> = gate_act.iter().zip(up.iter()).map(|(&g, &u)| g * u).collect();
            self.gpu_matmul_or_cpu(&ln("mlp.down_proj.weight"), &mid, t, ffn_inter, h, &mv_pc(ffn_inter, h))
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
            let h3b = f32_slice_to_bytes(&hidden3);
            unsafe { (*self.act_ptr_mut(ACT_FFIN)).write(&h3b).unwrap(); }  // reuse ACT_FFIN as PLE input
            let lpb = f32_slice_to_bytes(layer_ple);
            unsafe { (*self.act_ptr_mut(ACT_PLE_LAYER)).write(&lpb).unwrap(); }

            let inp_p   = self.act_ptr(ACT_FFIN);
            let pg_p    = self.act_ptr(ACT_PLE_G);
            let gelu_p  = self.act_ptr(ACT_PLE_GELU);
            let layer_p = self.act_ptr(ACT_PLE_LAYER);
            let mid_p   = self.act_ptr(ACT_PLE_MID);
            let pc_p    = self.act_ptr(ACT_PLE_C);

            let pgw = &self.gpu_weights[&ln("per_layer_input_gate.weight")] as *const compute::Buffer;
            let ppw = &self.gpu_weights[&ln("per_layer_projection.weight")] as *const compute::Buffer;

            let gelu_wg_ple = ((ple_dim + 511) / 512) as u32;
            let mul_wg_ple  = ((ple_dim + 255) / 256) as u32;
            let gelu_pc_ple = {
                use std::io::Write;
                let mut v = Vec::with_capacity(6 * 4);
                v.write_all(&(ple_dim as u32).to_le_bytes()).unwrap();
                v.write_all(&1u32.to_le_bytes()).unwrap();
                for _ in 0..4 { v.write_all(&0u32.to_le_bytes()).unwrap(); }
                v
            };
            let mul_pc_ple = {
                use std::io::Write;
                let n = ple_dim as u32;
                let mut v = Vec::with_capacity(29 * 4);
                for &x in &[n, n,1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                for &x in &[n, 1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                for &x in &[n, 1u32,1,1, 1u32,n,n,n] { v.write_all(&x.to_le_bytes()).unwrap(); }
                for &x in &[0u32, 0u32, 0u32, 0u32] { v.write_all(&x.to_le_bytes()).unwrap(); }
                v
            };

            let eng = self.engine.as_mut().unwrap();
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, shader, &[&*pgw, &*inp_p, &*pg_p], &mv_pc(h, ple_dim), (ple_dim as u32, t as u32, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "gelu_f32", &[&*pg_p, &*gelu_p], &gelu_pc_ple, (gelu_wg_ple, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, "mul_f32_f32_f32", &[&*gelu_p, &*layer_p, &*mid_p], &mul_pc_ple, (mul_wg_ple, 1, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, shader, &[&*ppw, &*mid_p, &*pc_p], &mv_pc(ple_dim, h), (h as u32, t as u32, 1)).unwrap();
            }
            eng.submit_batch(cb).unwrap();  // ONE fence wait for the whole PLE branch
            read_f32_buf(unsafe { &*pc_p }, t * h)
        } else {
            let pgw = self.inner.weights.f32_slice(&ln("per_layer_input_gate.weight"));
            let gate_ple = model::cpu_matmul(&hidden3, pgw, 1, h, ple_dim);
            let gate_ple_act = model::cpu_gelu(&gate_ple);
            let gated: Vec<f32> = gate_ple_act.iter().zip(layer_ple.iter())
                .map(|(&g, &p)| g * p).collect();
            let ppw = self.inner.weights.f32_slice(&ln("per_layer_projection.weight"));
            model::cpu_matmul(&gated, ppw, 1, ple_dim, h)
        };
        let contrib_normed = model::cpu_rms_norm(&contrib, &ple_norm_w, eps);
        hidden3.iter_mut().zip(contrib_normed.iter()).for_each(|(hv, &c)| *hv += c);

        // Layer scalar (pre-extracted)
        hidden3.iter_mut().for_each(|v| *v *= layer_scalar);
        if layer_idx == 0 { log::debug!("L{layer_idx} END: {}µs total", _t_layer.elapsed().as_micros()); }
        hidden3
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

    fn gpu_matmul_or_cpu(&mut self, weight_name: &str, x: &[f32],
                          t: usize, k: usize, n: usize, pc: &[u8]) -> Vec<f32> {
        if let (Some(eng), Some(w_ptr)) = (
            self.engine.as_mut(),
            self.gpu_weights.get(weight_name).map(|b| b as *const compute::Buffer)
        ) {
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let inp_p = &inp as *const compute::Buffer;
            let out_p = &out as *const compute::Buffer;
            let cb = eng.begin_batch().unwrap();
            unsafe {
                eng.record_to(cb, "mul_mat_vec_f16_f32_f32",
                    &[&*w_ptr, &*inp_p, &*out_p], pc, (n as u32, t as u32, 1)).unwrap();
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
            let xb = f32_slice_to_bytes(x);
            let inp = eng.alloc_host_coherent_storage((x.len() * 4) as u64).unwrap();
            inp.write(&xb).unwrap();
            let out1 = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let out2 = eng.alloc_host_coherent_storage((t * n * 4) as u64).unwrap();
            let inp_p  = &inp  as *const compute::Buffer;
            let out1_p = &out1 as *const compute::Buffer;
            let out2_p = &out2 as *const compute::Buffer;
            let cb = eng.begin_batch().unwrap();
            unsafe {
                let sh = "mul_mat_vec_f16_f32_f32";
                eng.record_to(cb, sh, &[&*w1_ptr, &*inp_p, &*out1_p], pc, (n as u32, t as u32, 1)).unwrap();
                eng.record_barrier_to(cb);
                eng.record_to(cb, sh, &[&*w2_ptr, &*inp_p, &*out2_p], pc, (n as u32, t as u32, 1)).unwrap();
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

/// Returns true if this weight tensor should be uploaded to GPU as f16.
/// Norm weights, scalars, and embeddings stay as f32 for precision.
fn is_matvec_weight(name: &str) -> bool {
    // Projection weights: q/k/v/o/gate/up/down, PLE gate, PLE projection
    name.ends_with("_proj.weight")          // e.g. q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        || name.ends_with("_gate.weight")   // e.g. per_layer_input_gate.weight
        || name.ends_with("_projection.weight")  // e.g. per_layer_projection.weight
        // embed_tokens.weight stays f32 (LM head precision), layernorm/scalar stay f32
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; data.len() * 4];
    for (i, &v) in data.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Convert f32 weights to f16 bytes for GPU upload.
/// f16 halves memory bandwidth which is the main bottleneck for matvec ops.
fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; data.len() * 2];
    for (i, &v) in data.iter().enumerate() {
        let h = half::f16::from_f32(v);
        bytes[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    bytes
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

    m.add_class::<VulkanDevice>()?;
    m.add_class::<VulkanContext>()?;
    m.add_class::<GpuTensor>()?;
    m.add_class::<VulkanModel>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__vulkan_available__", device::is_vulkan_available())?;

    Ok(())
}
