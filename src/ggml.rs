// SPDX-License-Identifier: Apache-2.0
//! Minimal ggml Vulkan compute bridge.
//!
//! This is intentionally tiny: it proves that Python can call into Rust,
//! Rust can call ggml, and ggml can execute a Vulkan `ggml_mul_mat`.

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

const GGML_DEFAULT_GRAPH_SIZE: usize = 2048;

// Keep this hand-written enum value in sync with the pinned ggml submodule
// when bumping it. Runtime validation below checks that it still resolves to
// ggml's "f32" tensor type; bindgen can replace this once the FFI surface grows.
const GGML_TYPE_F32: i32 = 0;

#[repr(C)]
struct GgmlInitParams {
    mem_size: usize,
    mem_buffer: *mut c_void,
    no_alloc: bool,
}

#[repr(C)]
struct GgmlContext {
    _private: [u8; 0],
}

#[repr(C)]
struct GgmlTensor {
    _private: [u8; 0],
}

#[repr(C)]
struct GgmlCGraph {
    _private: [u8; 0],
}

#[repr(C)]
struct GgmlBackend {
    _private: [u8; 0],
}

#[repr(C)]
struct GgmlBackendSched {
    _private: [u8; 0],
}

extern "C" {
    fn ggml_init(params: GgmlInitParams) -> *mut GgmlContext;
    fn ggml_free(ctx: *mut GgmlContext);
    fn ggml_tensor_overhead() -> usize;
    fn ggml_graph_overhead() -> usize;
    fn ggml_new_graph(ctx: *mut GgmlContext) -> *mut GgmlCGraph;
    fn ggml_new_tensor_2d(
        ctx: *mut GgmlContext,
        tensor_type: i32,
        ne0: i64,
        ne1: i64,
    ) -> *mut GgmlTensor;
    fn ggml_mul_mat(
        ctx: *mut GgmlContext,
        a: *mut GgmlTensor,
        b: *mut GgmlTensor,
    ) -> *mut GgmlTensor;
    fn ggml_build_forward_expand(graph: *mut GgmlCGraph, tensor: *mut GgmlTensor);
    fn ggml_nbytes(tensor: *const GgmlTensor) -> usize;
    fn ggml_type_name(tensor_type: i32) -> *const c_char;
    fn ggml_status_to_string(status: i32) -> *const c_char;

    fn ggml_backend_vk_init(dev_num: usize) -> *mut GgmlBackend;
    fn ggml_backend_vk_get_device_count() -> i32;
    fn ggml_backend_cpu_init() -> *mut GgmlBackend;
    fn ggml_backend_free(backend: *mut GgmlBackend);
    fn ggml_backend_tensor_set(
        tensor: *mut GgmlTensor,
        data: *const c_void,
        offset: usize,
        size: usize,
    );
    fn ggml_backend_tensor_get(
        tensor: *const GgmlTensor,
        data: *mut c_void,
        offset: usize,
        size: usize,
    );
    fn ggml_backend_name(backend: *mut GgmlBackend) -> *const c_char;
    fn ggml_backend_sched_new(
        backends: *mut *mut GgmlBackend,
        bufts: *mut c_void,
        n_backends: i32,
        graph_size: usize,
        parallel: bool,
        op_offload: bool,
    ) -> *mut GgmlBackendSched;
    fn ggml_backend_sched_free(sched: *mut GgmlBackendSched);
    fn ggml_backend_sched_reset(sched: *mut GgmlBackendSched);
    fn ggml_backend_sched_alloc_graph(sched: *mut GgmlBackendSched, graph: *mut GgmlCGraph)
        -> bool;
    fn ggml_backend_sched_get_tensor_backend(
        sched: *mut GgmlBackendSched,
        node: *mut GgmlTensor,
    ) -> *mut GgmlBackend;
    fn ggml_backend_sched_graph_compute(
        sched: *mut GgmlBackendSched,
        graph: *mut GgmlCGraph,
    ) -> i32;
}

struct Context(*mut GgmlContext);

impl Drop for Context {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ggml_free(self.0) };
        }
    }
}

struct Backend(*mut GgmlBackend);

impl Drop for Backend {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ggml_backend_free(self.0) };
        }
    }
}

struct Scheduler(*mut GgmlBackendSched);

impl Drop for Scheduler {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ggml_backend_sched_free(self.0) };
        }
    }
}

pub struct MatmulResult {
    pub output: Vec<f32>,
    pub backend_name: String,
}

pub fn vulkan_matmul_f32(
    a: &[f32],
    b_transposed: &[f32],
    m: usize,
    k: usize,
    n: usize,
    device_idx: usize,
) -> Result<MatmulResult, String> {
    validate_dims(a, b_transposed, m, k, n)?;

    unsafe {
        validate_ggml_ffi_constants()?;

        let device_count = ggml_backend_vk_get_device_count();
        if device_count <= 0 {
            return Err("ggml Vulkan backend found no Vulkan devices".to_string());
        }
        if device_idx >= device_count as usize {
            return Err(format!(
                "Vulkan device index {device_idx} out of range; ggml sees {device_count} device(s)"
            ));
        }

        let graph_buf_size = ggml_tensor_overhead()
            .checked_mul(GGML_DEFAULT_GRAPH_SIZE)
            .and_then(|size| size.checked_add(ggml_graph_overhead()))
            .ok_or_else(|| "ggml graph buffer size overflowed".to_string())?;
        let mut graph_buf = vec![0u8; graph_buf_size];

        let params = GgmlInitParams {
            mem_size: graph_buf.len(),
            mem_buffer: graph_buf.as_mut_ptr().cast(),
            no_alloc: true,
        };
        let ctx = Context(ggml_init(params));
        if ctx.0.is_null() {
            return Err("ggml_init failed".to_string());
        }

        let vk_backend = Backend(ggml_backend_vk_init(device_idx));
        if vk_backend.0.is_null() {
            return Err(format!(
                "ggml_backend_vk_init failed for device {device_idx}"
            ));
        }

        let cpu_backend = Backend(ggml_backend_cpu_init());
        if cpu_backend.0.is_null() {
            return Err("ggml CPU backend initialization failed".to_string());
        }

        let mut backends = [vk_backend.0, cpu_backend.0];
        let sched = Scheduler(ggml_backend_sched_new(
            backends.as_mut_ptr(),
            ptr::null_mut(),
            backends.len() as i32,
            GGML_DEFAULT_GRAPH_SIZE,
            false,
            true,
        ));
        if sched.0.is_null() {
            return Err("ggml backend scheduler initialization failed".to_string());
        }

        // To compute Python-style `A @ B`, ggml receives `B^T` as the weight
        // tensor and `A` as the input tensor. With shapes `(m, k) @ (k, n)`,
        // ggml tensors are weight `(k, n)`, input `(k, m)`, output `(m, n)`.
        let weight = ggml_new_tensor_2d(ctx.0, GGML_TYPE_F32, k as i64, n as i64);
        let input = ggml_new_tensor_2d(ctx.0, GGML_TYPE_F32, k as i64, m as i64);

        if weight.is_null() || input.is_null() {
            return Err("ggml tensor allocation failed".to_string());
        }

        let result = ggml_mul_mat(ctx.0, weight, input);
        if result.is_null() {
            return Err("ggml_mul_mat failed to create an output tensor".to_string());
        }

        let graph = ggml_new_graph(ctx.0);
        if graph.is_null() {
            return Err("ggml_new_graph failed".to_string());
        }

        ggml_build_forward_expand(graph, result);
        ggml_backend_sched_reset(sched.0);

        if !ggml_backend_sched_alloc_graph(sched.0, graph) {
            return Err("ggml failed to allocate the Vulkan graph".to_string());
        }
        let backend_name = tensor_backend_name(sched.0, result)
            .ok_or_else(|| "ggml did not assign the matmul result to a backend".to_string())?;

        ggml_backend_tensor_set(
            weight,
            b_transposed.as_ptr().cast(),
            0,
            std::mem::size_of_val(b_transposed),
        );
        ggml_backend_tensor_set(input, a.as_ptr().cast(), 0, std::mem::size_of_val(a));

        let status = ggml_backend_sched_graph_compute(sched.0, graph);
        if !is_success_status(status) {
            return Err(format!(
                "ggml graph compute failed: {}",
                status_name(status)
            ));
        }

        let result_bytes = ggml_nbytes(result);
        let expected_bytes = m
            .checked_mul(n)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "ggml output size overflowed".to_string())?;

        if result_bytes != expected_bytes {
            return Err(format!(
                "ggml output size mismatch: expected {expected_bytes} bytes, got {result_bytes}"
            ));
        }

        let mut output = vec![0.0f32; m * n];
        ggml_backend_tensor_get(result, output.as_mut_ptr().cast(), 0, result_bytes);

        Ok(MatmulResult {
            output,
            backend_name,
        })
    }
}

fn validate_ggml_ffi_constants() -> Result<(), String> {
    unsafe {
        let type_name = ggml_type_name(GGML_TYPE_F32);
        if type_name.is_null() {
            return Err("ggml_type_name returned null for GGML_TYPE_F32".to_string());
        }
        let type_name = CStr::from_ptr(type_name).to_string_lossy();
        if type_name != "f32" {
            return Err(format!(
                "GGML_TYPE_F32 binding mismatch: expected 'f32', got '{type_name}'"
            ));
        }
        Ok(())
    }
}

fn tensor_backend_name(sched: *mut GgmlBackendSched, tensor: *mut GgmlTensor) -> Option<String> {
    unsafe {
        let backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        if backend.is_null() {
            return None;
        }
        let name = ggml_backend_name(backend);
        if name.is_null() {
            return None;
        }
        Some(CStr::from_ptr(name).to_string_lossy().into_owned())
    }
}

fn validate_dims(
    a: &[f32],
    b_transposed: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), String> {
    if m == 0 || k == 0 || n == 0 {
        return Err("matmul dimensions must be non-zero".to_string());
    }
    if a.len() != m * k {
        return Err(format!(
            "left input has {} values, expected {} for shape ({m}, {k})",
            a.len(),
            m * k
        ));
    }
    if b_transposed.len() != n * k {
        return Err(format!(
            "right input has {} transposed values, expected {} for original shape ({k}, {n})",
            b_transposed.len(),
            n * k
        ));
    }
    if m > i64::MAX as usize || k > i64::MAX as usize || n > i64::MAX as usize {
        return Err("matmul dimensions exceed ggml i64 tensor limits".to_string());
    }
    Ok(())
}

fn is_success_status(status: i32) -> bool {
    status_name(status) == "GGML status: success"
}

fn status_name(status: i32) -> String {
    unsafe {
        let ptr = ggml_status_to_string(status);
        if ptr.is_null() {
            return status.to_string();
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
