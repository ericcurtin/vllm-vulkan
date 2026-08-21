// SPDX-License-Identifier: Apache-2.0
//! `VulkanContext`/`GpuTensor` — the Python-accessible low-level compute
//! context (raw execute/execute_batch/execute_chained + tensor upload API).
//! Extracted verbatim from lib.rs (M1). Independent unit: `VulkanModel`
//! never touches these.

use crate::compute;
use crate::device;
use crate::include_all_shaders;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
            dev.caps(),
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
        // BufRef mirrors execute_batch/execute_mixed: `bytes`-derived temporaries
        // live in `temp_bufs` and are referenced by index (never by a raw pointer
        // to a moved stack local — that was UB). GpuTensor buffers and the
        // intermediate/output buffers (which outlive recording) use a stable Ptr.
        enum BufRef { Temp(usize), Ptr(*const compute::Buffer) }
        let mut temp_bufs: Vec<compute::Buffer> = Vec::new();

        // Resolve bindings for Op 0.
        let mut refs0: Vec<BufRef> = Vec::new();
        for binding in &bindings0 {
            Python::with_gil(|py_inner| -> PyResult<()> {
                if let Ok(gt) = binding.downcast_bound::<GpuTensor>(py_inner) {
                    refs0.push(BufRef::Ptr(&gt.borrow().buf as *const compute::Buffer));
                } else if let Ok(bytes) = binding.downcast_bound::<pyo3::types::PyBytes>(py_inner) {
                    let data = bytes.as_bytes();
                    let buf = self.engine.alloc_host_coherent_storage(data.len() as u64)
                        .map_err(PyRuntimeError::new_err)?;
                    buf.write(data).map_err(PyRuntimeError::new_err)?;
                    let idx = temp_bufs.len();
                    temp_bufs.push(buf);
                    refs0.push(BufRef::Temp(idx));
                } else {
                    return Err(PyRuntimeError::new_err("binding must be GpuTensor or bytes"));
                }
                Ok(())
            })?;
        }

        // Intermediate buffer: Op 0 writes here, Op 1 reads here.
        let inter_buf = self.engine.alloc_host_coherent_storage(output_size0)
            .map_err(PyRuntimeError::new_err)?;
        refs0.push(BufRef::Ptr(&inter_buf as *const compute::Buffer));

        // Op 1 output.
        let out1_buf = self.engine.alloc_host_coherent_storage(output_size1)
            .map_err(PyRuntimeError::new_err)?;

        // Resolve bindings for Op 1 (NOT including the intermediate — added below).
        let mut refs1: Vec<BufRef> = Vec::new();
        for binding in &bindings1 {
            Python::with_gil(|py_inner| -> PyResult<()> {
                if let Ok(gt) = binding.downcast_bound::<GpuTensor>(py_inner) {
                    refs1.push(BufRef::Ptr(&gt.borrow().buf as *const compute::Buffer));
                } else if let Ok(bytes) = binding.downcast_bound::<pyo3::types::PyBytes>(py_inner) {
                    let data = bytes.as_bytes();
                    let buf = self.engine.alloc_host_coherent_storage(data.len() as u64)
                        .map_err(PyRuntimeError::new_err)?;
                    buf.write(data).map_err(PyRuntimeError::new_err)?;
                    let idx = temp_bufs.len();
                    temp_bufs.push(buf);
                    refs1.push(BufRef::Temp(idx));
                } else {
                    return Err(PyRuntimeError::new_err("binding must be GpuTensor or bytes"));
                }
                Ok(())
            })?;
        }
        // Append inter_buf as input to Op 1, then out1_buf as output.
        refs1.push(BufRef::Ptr(&inter_buf as *const compute::Buffer));
        refs1.push(BufRef::Ptr(&out1_buf as *const compute::Buffer));

        // ── Record both dispatches into one command buffer ────────────────
        let cb = self.engine.begin_batch().map_err(PyRuntimeError::new_err)?;

        // Op 0: resolve BufRefs to &Buffer (temp_bufs borrowed here, disjoint from self.engine).
        {
            let buf_refs0: Vec<&compute::Buffer> = refs0.iter()
                .map(|r| match r {
                    BufRef::Temp(i) => &temp_bufs[*i],
                    BufRef::Ptr(p) => unsafe { &**p },
                })
                .collect();
            self.engine.record_to(cb, shader0, &buf_refs0, &pc0, wg0)
                .map_err(PyRuntimeError::new_err)?;
            self.engine.record_barrier_to(cb);  // Op 1 reads Op 0's output
        }

        // Op 1.
        {
            let buf_refs1: Vec<&compute::Buffer> = refs1.iter()
                .map(|r| match r {
                    BufRef::Temp(i) => &temp_bufs[*i],
                    BufRef::Ptr(p) => unsafe { &**p },
                })
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
