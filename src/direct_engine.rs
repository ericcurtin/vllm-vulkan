//! M0 — reusable direct-submit compute backend (libdrm `amdgpu_cs`, no RADV).
//!
//! Productionizes the `spike/s1a-direct-submit/{stepA,s3,s4mlx4}` proof into a
//! Rust backend that lives BESIDE `compute::ComputeEngine` and routes decode-hot
//! dispatches through the kernel `amdgpu_cs` ioctl (~18 µs/submit) instead of
//! RADV `vkQueueSubmit`+fence (~478 µs). Gated by `VLLM_VULKAN_DIRECT_SUBMIT`.
//!
//! This file is the shared, model-independent infra (rollout milestone M0):
//!   - [`DirectDevice`]     persistent amdgpu device + compute context (libdrm dlopen'd).
//!   - [`Bo`] / arena       BO alloc + VA map + CPU map; resident VAs registered once.
//!   - [`RegisteredKernel`] a gfx1013 code object uploaded to VRAM + its descriptors.
//!   - [`IbBuilder`]        multi-dispatch IB builder with the proven inter-dispatch
//!                          barrier (`CS_PARTIAL_FLUSH` + `ACQUIRE_MEM`, P1-validated).
//!   - [`DirectDevice::submit_wait`]  one `amdgpu_cs_submit` + seq_no fence per IB.
//!
//! SAFETY: sanctioned libdrm ioctl path ONLY — no amdgpu_regs2/debugfs writes.
//!
//! The M0 gate ([`debug_direct_engine_replay`]) replays the S3 f32 matvec, the
//! s4mlx4 4-bit matvec, and a 2-dispatch barriered chain through THIS engine and
//! checks each bit-exact vs a double-precision CPU reference (the same ground
//! truth the engine's own Vulkan kernels are gated against).

#![allow(clippy::missing_safety_doc)]

use crate::direct_kernels::{KernelDef, MATVEC_F32, MLX4_MV, MLX4_MV_GROUP_SIZE, MLX4_MV_NUM_ROWS};
use libloading::Library;
use std::ffi::c_void;
use std::os::raw::c_int;

// ── libdrm ABI (verbatim from spike/s1a-direct-submit/vendor/amdgpu.h) ──────────
type DeviceHandle = *mut c_void;
type ContextHandle = *mut c_void;
type BoHandle = *mut c_void;
type BoListHandle = *mut c_void;
type VaHandle = *mut c_void;

const AMDGPU_GEM_DOMAIN_GTT: u32 = 0x2;
const AMDGPU_GEM_DOMAIN_VRAM: u32 = 0x4;
const AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED: u64 = 1 << 0;
const AMDGPU_HW_IP_COMPUTE: u32 = 1;
const AMDGPU_VA_OP_MAP: u32 = 1;
const AMDGPU_VM_PAGE_READABLE: u64 = 1 << 1;
const AMDGPU_VM_PAGE_WRITEABLE: u64 = 1 << 2;
const AMDGPU_VM_PAGE_EXECUTABLE: u64 = 1 << 3;
const AMDGPU_TIMEOUT_INFINITE: u64 = u64::MAX;
const GPU_VA_RANGE_GENERAL: u32 = 0;

#[repr(C)]
struct BoAllocRequest {
    alloc_size: u64,
    phys_alignment: u64,
    preferred_heap: u32,
    flags: u64,
}
#[repr(C)]
struct CsIbInfo {
    flags: u64,
    ib_mc_address: u64,
    size: u32,
}
#[repr(C)]
struct CsFenceInfo {
    handle: BoHandle,
    offset: u64,
}
#[repr(C)]
struct CsRequest {
    flags: u64,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
    resources: BoListHandle,
    number_of_dependencies: u32,
    dependencies: *mut c_void,
    number_of_ibs: u32,
    ibs: *mut CsIbInfo,
    seq_no: u64,
    fence_info: CsFenceInfo,
}
#[repr(C)]
struct CsFence {
    context: ContextHandle,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
    fence: u64,
}

type FnDevInit = unsafe extern "C" fn(c_int, *mut u32, *mut u32, *mut DeviceHandle) -> c_int;
type FnDevDeinit = unsafe extern "C" fn(DeviceHandle) -> c_int;
type FnBoAlloc = unsafe extern "C" fn(DeviceHandle, *const BoAllocRequest, *mut BoHandle) -> c_int;
type FnBoCpuMap = unsafe extern "C" fn(BoHandle, *mut *mut c_void) -> c_int;
type FnVaRangeAlloc = unsafe extern "C" fn(
    DeviceHandle,
    u32,
    u64,
    u64,
    u64,
    *mut u64,
    *mut VaHandle,
    u64,
) -> c_int;
type FnBoVaOpRaw =
    unsafe extern "C" fn(DeviceHandle, BoHandle, u64, u64, u64, u64, u32) -> c_int;
type FnBoListCreate =
    unsafe extern "C" fn(DeviceHandle, u32, *mut BoHandle, *mut u8, *mut BoListHandle) -> c_int;
type FnCtxCreate = unsafe extern "C" fn(DeviceHandle, *mut ContextHandle) -> c_int;
type FnCtxFree = unsafe extern "C" fn(ContextHandle) -> c_int;
type FnCsSubmit = unsafe extern "C" fn(ContextHandle, u64, *mut CsRequest, u32) -> c_int;
type FnCsFenceStatus = unsafe extern "C" fn(*mut CsFence, u64, u64, *mut u32) -> c_int;

struct Drm {
    _lib: Library, // keep the code mapped; the fn pointers below reference into it
    dev_init: FnDevInit,
    dev_deinit: FnDevDeinit,
    bo_alloc: FnBoAlloc,
    bo_cpu_map: FnBoCpuMap,
    va_range_alloc: FnVaRangeAlloc,
    bo_va_op_raw: FnBoVaOpRaw,
    bo_list_create: FnBoListCreate,
    ctx_create: FnCtxCreate,
    ctx_free: FnCtxFree,
    cs_submit: FnCsSubmit,
    cs_fence_status: FnCsFenceStatus,
}

impl Drm {
    fn load() -> Result<Drm, String> {
        // Try the SONAME then the RADV-linked absolute path (matches the C harness).
        let lib = unsafe { Library::new("libdrm_amdgpu.so.1") }
            .or_else(|_| unsafe { Library::new("/usr/lib64/libdrm_amdgpu.so.1") })
            .map_err(|e| format!("dlopen libdrm_amdgpu.so.1: {e}"))?;
        macro_rules! sym {
            ($t:ty, $n:literal) => {
                *unsafe { lib.get::<$t>($n) }.map_err(|e| {
                    format!("dlsym {}: {e}", String::from_utf8_lossy(&$n[..$n.len() - 1]))
                })?
            };
        }
        let d = Drm {
            dev_init: sym!(FnDevInit, b"amdgpu_device_initialize\0"),
            dev_deinit: sym!(FnDevDeinit, b"amdgpu_device_deinitialize\0"),
            bo_alloc: sym!(FnBoAlloc, b"amdgpu_bo_alloc\0"),
            bo_cpu_map: sym!(FnBoCpuMap, b"amdgpu_bo_cpu_map\0"),
            va_range_alloc: sym!(FnVaRangeAlloc, b"amdgpu_va_range_alloc\0"),
            bo_va_op_raw: sym!(FnBoVaOpRaw, b"amdgpu_bo_va_op_raw\0"),
            bo_list_create: sym!(FnBoListCreate, b"amdgpu_bo_list_create\0"),
            ctx_create: sym!(FnCtxCreate, b"amdgpu_cs_ctx_create\0"),
            ctx_free: sym!(FnCtxFree, b"amdgpu_cs_ctx_free\0"),
            cs_submit: sym!(FnCsSubmit, b"amdgpu_cs_submit\0"),
            cs_fence_status: sym!(FnCsFenceStatus, b"amdgpu_cs_query_fence_status\0"),
            _lib: lib,
        };
        Ok(d)
    }
}

// ── PM4 constants (gfx10.1 / GC 10.1.3) ─────────────────────────────────────────
const PACKET_TYPE3: u32 = 3;
const fn packet3(op: u32, n: u32) -> u32 {
    (PACKET_TYPE3 << 30) | ((op & 0xFF) << 8) | ((n & 0x3FFF) << 16)
}
const fn packet3_compute(op: u32, n: u32) -> u32 {
    packet3(op, n) | (1 << 1)
}
const PKT3_SET_SH_REG: u32 = 0x76;
const PKT3_SET_SH_REG_INDEX: u32 = 0x9B;
const PKT3_SET_UCONFIG_REG: u32 = 0x79;
const PKT3_DISPATCH_DIRECT: u32 = 0x15;
const PKT3_EVENT_WRITE: u32 = 0x46;
const PKT3_ACQUIRE_MEM: u32 = 0x58;
const GFX_COMPUTE_NOP: u32 = 0xffff1000;
const SH_REG_BASE_GFX10: u32 = 0x2C00;
const S_CODE_END: u32 = 0xBF9F0000;
const EV_CS_PARTIAL_FLUSH: u32 = 7 | (4 << 8); // CS_PARTIAL_FLUSH event
const GCR_FULL: u32 = 0x0001_C3F0; // ACQUIRE_MEM GCR_CNTL (P1-proven full cache coherence)

/// A GPU buffer object: BO handle + a mapped GPU VA (`mc`) + host-mapped pointer.
struct Bo {
    handle: BoHandle,
    _va: VaHandle,
    mc: u64,
    cpu: *mut c_void,
    size: u64,
}
impl Bo {
    unsafe fn as_u32(&self) -> *mut u32 {
        self.cpu as *mut u32
    }
    unsafe fn write_f32(&self, data: &[f32]) {
        std::ptr::copy_nonoverlapping(data.as_ptr(), self.cpu as *mut f32, data.len());
    }
    unsafe fn write_u32(&self, data: &[u32]) {
        std::ptr::copy_nonoverlapping(data.as_ptr(), self.cpu as *mut u32, data.len());
    }
    unsafe fn read_f32(&self, n: usize) -> Vec<f32> {
        let mut v = vec![0f32; n];
        std::ptr::copy_nonoverlapping(self.cpu as *const f32, v.as_mut_ptr(), n);
        v
    }
}

/// A code object uploaded to VRAM + the descriptor registers to bind it.
struct RegisteredKernel {
    shader_va: u64,
    rsrc1: u32,
    rsrc2: u32,
    rsrc3: u32,
    block: u32,
}

/// Persistent direct-submit device: amdgpu device + compute context, kept alive
/// for the process. Coexists with RADV's queue (separate context, same rings).
pub struct DirectDevice {
    drm: Drm,
    fd: c_int,
    dev: DeviceHandle,
    ctx: ContextHandle,
    // Every BO allocated through this device; owns them for the lifetime of the
    // device so their VAs stay mapped and they survive into the bo_list.
    bos: Vec<Bo>,
    // Arena index of the most-recently-registered kernel's VRAM code BO.
    last_kernel_bo: Option<usize>,
}

impl DirectDevice {
    /// Whether the direct-submit backend is flag-enabled (`VLLM_VULKAN_DIRECT_SUBMIT=1`).
    pub fn flag_enabled() -> bool {
        matches!(std::env::var("VLLM_VULKAN_DIRECT_SUBMIT").as_deref(), Ok("1"))
    }

    pub fn open(render_node: &str) -> Result<DirectDevice, String> {
        let drm = Drm::load()?;
        let cpath = std::ffi::CString::new(render_node).unwrap();
        let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(format!("open {render_node}: {}", std::io::Error::last_os_error()));
        }
        let mut dev: DeviceHandle = std::ptr::null_mut();
        let (mut maj, mut min) = (0u32, 0u32);
        if unsafe { (drm.dev_init)(fd, &mut maj, &mut min, &mut dev) } != 0 {
            unsafe { libc::close(fd) };
            return Err("amdgpu_device_initialize failed".into());
        }
        let mut ctx: ContextHandle = std::ptr::null_mut();
        if unsafe { (drm.ctx_create)(dev, &mut ctx) } != 0 {
            return Err("amdgpu_cs_ctx_create failed".into());
        }
        Ok(DirectDevice { drm, fd, dev, ctx, bos: Vec::new(), last_kernel_bo: None })
    }

    /// Allocate a BO, map a GPU VA over it, and CPU-map it. Returns an index into
    /// the arena (`self.bos`) so the buffer stays resident for the device's life.
    fn alloc(&mut self, size: u64, heap: u32, flags: u64) -> Result<usize, String> {
        let size = (size + 4095) & !4095u64;
        let req = BoAllocRequest {
            alloc_size: size,
            phys_alignment: 4096,
            preferred_heap: heap,
            flags,
        };
        let mut handle: BoHandle = std::ptr::null_mut();
        if unsafe { (self.drm.bo_alloc)(self.dev, &req, &mut handle) } != 0 {
            return Err(format!("amdgpu_bo_alloc({size})"));
        }
        let mut mc = 0u64;
        let mut va: VaHandle = std::ptr::null_mut();
        if unsafe {
            (self.drm.va_range_alloc)(self.dev, GPU_VA_RANGE_GENERAL, size, 4096, 0, &mut mc, &mut va, 0)
        } != 0
        {
            return Err("amdgpu_va_range_alloc".into());
        }
        if unsafe {
            (self.drm.bo_va_op_raw)(
                self.dev,
                handle,
                0,
                size,
                mc,
                AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE,
                AMDGPU_VA_OP_MAP,
            )
        } != 0
        {
            return Err("amdgpu_bo_va_op_raw MAP".into());
        }
        let mut cpu: *mut c_void = std::ptr::null_mut();
        if unsafe { (self.drm.bo_cpu_map)(handle, &mut cpu) } != 0 {
            return Err("amdgpu_bo_cpu_map".into());
        }
        self.bos.push(Bo { handle, _va: va, mc, cpu, size });
        Ok(self.bos.len() - 1)
    }

    fn gtt(&mut self, size: u64) -> Result<usize, String> {
        self.alloc(size, AMDGPU_GEM_DOMAIN_GTT, 0)
    }

    /// Upload a kernel's code object into a fresh VRAM BO (256-aligned base, tail
    /// padded with `s_code_end`) and return its bindable descriptors.
    fn register_kernel(&mut self, k: &KernelDef) -> Result<RegisteredKernel, String> {
        let bytes = (k.text.len() * 4) as u64 + 256;
        let idx = self.alloc(bytes, AMDGPU_GEM_DOMAIN_VRAM, AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED)?;
        self.last_kernel_bo = Some(idx);
        let bo = &self.bos[idx];
        unsafe {
            let p = bo.as_u32();
            let words = (bo.size / 4) as usize;
            for w in 0..words {
                *p.add(w) = S_CODE_END;
            }
            std::ptr::copy_nonoverlapping(k.text.as_ptr(), p, k.text.len());
        }
        Ok(RegisteredKernel {
            shader_va: bo.mc,
            rsrc1: k.rsrc1,
            rsrc2: k.rsrc2,
            rsrc3: k.rsrc3,
            block: k.block,
        })
    }

    /// Submit one IB and wait on the kernel seq_no fence. `bo_idxs` is every BO the
    /// IB touches (code + kernargs + data + the cmd BO itself).
    fn submit_wait(&mut self, cmd_idx: usize, dwords: u32, bo_idxs: &[usize]) -> Result<(), String> {
        let mut handles: Vec<BoHandle> = bo_idxs.iter().map(|&i| self.bos[i].handle).collect();
        let mut bl: BoListHandle = std::ptr::null_mut();
        if unsafe {
            (self.drm.bo_list_create)(
                self.dev,
                handles.len() as u32,
                handles.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut bl,
            )
        } != 0
        {
            return Err("amdgpu_bo_list_create".into());
        }
        let mut ib = CsIbInfo { flags: 0, ib_mc_address: self.bos[cmd_idx].mc, size: dwords };
        let mut rq = CsRequest {
            flags: 0,
            ip_type: AMDGPU_HW_IP_COMPUTE,
            ip_instance: 0,
            ring: 0,
            resources: bl,
            number_of_dependencies: 0,
            dependencies: std::ptr::null_mut(),
            number_of_ibs: 1,
            ibs: &mut ib,
            seq_no: 0,
            fence_info: CsFenceInfo { handle: std::ptr::null_mut(), offset: 0 },
        };
        if unsafe { (self.drm.cs_submit)(self.ctx, 0, &mut rq, 1) } != 0 {
            return Err("amdgpu_cs_submit".into());
        }
        let mut f = CsFence {
            context: self.ctx,
            ip_type: AMDGPU_HW_IP_COMPUTE,
            ip_instance: 0,
            ring: 0,
            fence: rq.seq_no,
        };
        let mut expired = 0u32;
        if unsafe {
            (self.drm.cs_fence_status)(&mut f, AMDGPU_TIMEOUT_INFINITE, 0, &mut expired)
        } != 0
            || expired == 0
        {
            return Err("amdgpu_cs_query_fence_status".into());
        }
        Ok(())
    }
}

impl Drop for DirectDevice {
    fn drop(&mut self) {
        // BOs leak (freed by process teardown / kernel on fd close); free ctx+dev.
        unsafe {
            if !self.ctx.is_null() {
                (self.drm.ctx_free)(self.ctx);
            }
            if !self.dev.is_null() {
                (self.drm.dev_deinit)(self.dev);
            }
            if self.fd >= 0 {
                libc::close(self.fd);
            }
        }
    }
}

/// Builds a PM4 command stream: per-dispatch shader-state + DISPATCH_DIRECT, with
/// the P1-proven `CS_PARTIAL_FLUSH`+`ACQUIRE_MEM` barrier between dependent dispatches.
struct IbBuilder {
    w: Vec<u32>,
}
impl IbBuilder {
    fn new() -> Self {
        IbBuilder { w: Vec::with_capacity(256) }
    }
    /// Re-emit shader state (PGM_LO/HI, RSRC1/2/3, NUM_THREAD_X) for a code object.
    fn state(&mut self, k: &RegisteredKernel) {
        let p = &mut self.w;
        let sv = k.shader_va;
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 3), 0x204, 0, 0, 0]);
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 1), 0x218, 0]);
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 1), 0x22a, 0]);
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 6), 0x222, 0, 0, 0, 0, 0, 0]);
        p.extend_from_slice(&[packet3(PKT3_SET_UCONFIG_REG, 1), 0x7b, 0x20]);
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG_INDEX, 2), 0x30000216, 0xffffffff, 0xffffffff]);
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG_INDEX, 2), 0x30000219, 0xffffffff, 0xffffffff]);
        p.extend_from_slice(&[
            packet3_compute(PKT3_SET_SH_REG, 2),
            0x20c,
            (sv >> 8) as u32,
            (sv >> 40) as u32,
        ]);
        let regs = [
            (0x2e12 - SH_REG_BASE_GFX10, k.rsrc1),
            (0x2e13 - SH_REG_BASE_GFX10, k.rsrc2),
            (0x2e07 - SH_REG_BASE_GFX10, k.block),
            (0x2e08 - SH_REG_BASE_GFX10, 1),
            (0x2e09 - SH_REG_BASE_GFX10, 1),
        ];
        for (off, val) in regs {
            p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 1), off, val]);
        }
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 1), 0x228, k.rsrc3]);
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 1), 0x215, 0]);
    }
    /// Bind the kernarg pointer and launch `grid_x` workgroups.
    fn dispatch(&mut self, kernarg_mc: u64, grid_x: u32) {
        let p = &mut self.w;
        p.extend_from_slice(&[packet3_compute(PKT3_SET_SH_REG, 6), 0x240, 0, 0, 0, 0]);
        p.push(kernarg_mc as u32);
        p.push((kernarg_mc >> 32) as u32);
        p.extend_from_slice(&[packet3_compute(PKT3_DISPATCH_DIRECT, 3), grid_x, 1, 1, 1]);
    }
    /// Inter-dispatch barrier: wait for the prior dispatch's waves to drain
    /// (`CS_PARTIAL_FLUSH`) and flush+invalidate caches (`ACQUIRE_MEM`, GCR_CNTL
    /// full) — P1-proven necessary+sufficient for RAW between same-queue dispatches.
    fn barrier(&mut self) {
        let p = &mut self.w;
        p.extend_from_slice(&[packet3_compute(PKT3_EVENT_WRITE, 0), EV_CS_PARTIAL_FLUSH]);
        p.extend_from_slice(&[
            packet3_compute(PKT3_ACQUIRE_MEM, 6),
            0,
            0xffffffff,
            0x00ffffff,
            0,
            0,
            0x0000000A,
            GCR_FULL,
        ]);
    }
    /// Pad to a multiple of 8 dwords with COMPUTE NOPs; return dword count.
    fn finish(mut self) -> Vec<u32> {
        while self.w.len() & 7 != 0 {
            self.w.push(GFX_COMPUTE_NOP);
        }
        self.w
    }
}

// ── M0 replay gate ──────────────────────────────────────────────────────────────

fn cos_argmax(got: &[f32], want: &[f64]) -> (f64, usize, usize, f64) {
    let (mut dot, mut ng, mut nc, mut maxe) = (0f64, 0f64, 0f64, 0f64);
    let (mut amg, mut amw) = (0usize, 0usize);
    let (mut bg, mut bw) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let g = g as f64;
        dot += g * w;
        ng += g * g;
        nc += w * w;
        maxe = maxe.max((g - w).abs());
        if g > bg {
            bg = g;
            amg = i;
        }
        if w > bw {
            bw = w;
            amw = i;
        }
    }
    (dot / (ng.sqrt() * nc.sqrt() + 1e-30), amg, amw, maxe)
}

/// xorshift64 → f32 in [-1,1); matches the spike harnesses' deterministic input.
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        ((self.u() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

/// REPLAY 1 — S3 f32 matvec (branchless) through DirectEngine, single-dispatch IB.
/// out[r] = sum_k W[r*K+k] * x[k]. Gate vs double CPU ref.
unsafe fn replay_s3_matvec(dd: &mut DirectDevice, k: usize, m: usize) -> Result<String, String> {
    let mut rng = Rng(0xC0FFEE_1234567);
    let x: Vec<f32> = (0..k).map(|_| rng.f()).collect();
    let w: Vec<f32> = (0..m * k).map(|_| rng.f()).collect();

    let reg = dd.register_kernel(&MATVEC_F32)?;
    let wbo = dd.gtt((m * k * 4) as u64)?;
    let xbo = dd.gtt((k * 4) as u64)?;
    let obo = dd.gtt((m * 4) as u64)?;
    let kabo = dd.gtt(4096)?;
    let cmdbo = dd.gtt(4096)?;
    dd.bos[wbo].write_f32(&w);
    dd.bos[xbo].write_f32(&x);
    // kernarg: W@0 x@8 out@16 K@24
    let ka = dd.bos[kabo].as_u32();
    *(ka as *mut u64).add(0) = dd.bos[wbo].mc;
    *(ka as *mut u64).add(1) = dd.bos[xbo].mc;
    *(ka as *mut u64).add(2) = dd.bos[obo].mc;
    *ka.add(6) = k as u32;

    let mut ib = IbBuilder::new();
    ib.state(&reg);
    ib.dispatch(dd.bos[kabo].mc, (m / 64) as u32);
    let words = ib.finish();
    dd.bos[cmdbo].write_u32(&words);
    dd.submit_wait(cmdbo, words.len() as u32, &[reg_bo(dd), wbo, xbo, obo, kabo, cmdbo])?;

    let got = dd.bos[obo].read_f32(m);
    let mut refv = vec![0f64; m];
    for r in 0..m {
        let mut acc = 0f64;
        for c in 0..k {
            acc += w[r * k + c] as f64 * x[c] as f64;
        }
        refv[r] = acc;
    }
    let (cos, amg, amw, maxe) = cos_argmax(&got, &refv);
    let ok = cos >= 0.99999 && amg == amw;
    Ok(format!(
        "S3 f32 matvec  K={k} M={m}: cos={cos:.10} argmax {amg}=={amw} max_abs_err={maxe:.3e} -> {}",
        if ok { "PASS" } else { "*** FAIL ***" }
    ))
}

/// The most-recently-registered kernel's VRAM code BO index (tracked by
/// `register_kernel`) — it must be included in the submit's bo_list.
fn reg_bo(dd: &DirectDevice) -> usize {
    dd.last_kernel_bo.expect("register_kernel must run before submit")
}

/// REPLAY 2 — s4mlx4 4-bit matvec through DirectEngine, single-dispatch IB.
/// Optional `cap` dir loads a live Vulkan capture ({packed,scales,biases,x,vk_out}.bin).
unsafe fn replay_mlx4(
    dd: &mut DirectDevice,
    k: usize,
    n: usize,
    cap: Option<&str>,
) -> Result<String, String> {
    let gs = MLX4_MV_GROUP_SIZE as usize;
    let groups = k / gs;
    let wpr = k / 8;
    let (packed, scales, biases, x, vk): (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Option<Vec<f32>>) =
        if let Some(dir) = cap {
            let rd = |n: &str| -> Result<Vec<u8>, String> {
                std::fs::read(format!("{dir}/{n}")).map_err(|e| format!("{dir}/{n}: {e}"))
            };
            let packed: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&rd("packed.bin")?).to_vec();
            let scales: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rd("scales.bin")?).to_vec();
            let biases: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rd("biases.bin")?).to_vec();
            let x: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rd("x.bin")?).to_vec();
            let vk: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&rd("vk_out.bin")?).to_vec();
            (packed, scales, biases, x, Some(vk))
        } else {
            let mut rng = Rng(1234u64.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
            let mut packed = vec![0u32; n * wpr];
            let mut scales = vec![0f32; n * groups];
            let mut biases = vec![0f32; n * groups];
            for r in 0..n {
                for g in 0..groups {
                    let sc = 0.05 + ((rng.u() >> 40) as f32 / (1u32 << 24) as f32) * 0.2;
                    let bi = rng.f() * 0.1;
                    scales[r * groups + g] = sc;
                    biases[r * groups + g] = bi;
                }
                for j in 0..k {
                    let q = (rng.u() % 16) as u32;
                    packed[r * wpr + j / 8] |= q << ((j % 8) * 4);
                }
            }
            let x: Vec<f32> = (0..k).map(|_| rng.f()).collect();
            (packed, scales, biases, x, None)
        };

    let reg = dd.register_kernel(&MLX4_MV)?;
    let kbo = reg_bo(dd);
    let wbo = dd.gtt((packed.len() * 4) as u64)?;
    let sbo = dd.gtt((scales.len() * 4) as u64)?;
    let bbo = dd.gtt((biases.len() * 4) as u64)?;
    let xbo = dd.gtt((x.len() * 4) as u64)?;
    let obo = dd.gtt((n * 4) as u64)?;
    let kabo = dd.gtt(4096)?;
    let cmdbo = dd.gtt(4096)?;
    dd.bos[wbo].write_u32(&packed);
    dd.bos[sbo].write_f32(&scales);
    dd.bos[bbo].write_f32(&biases);
    dd.bos[xbo].write_f32(&x);
    // kernarg: W@0 sc@8 bi@16 x@24 dst@32 | k@40 n@44 gs@48 poff@52 sboff@56
    let ka = dd.bos[kabo].as_u32();
    *(ka as *mut u64).add(0) = dd.bos[wbo].mc;
    *(ka as *mut u64).add(1) = dd.bos[sbo].mc;
    *(ka as *mut u64).add(2) = dd.bos[bbo].mc;
    *(ka as *mut u64).add(3) = dd.bos[xbo].mc;
    *(ka as *mut u64).add(4) = dd.bos[obo].mc;
    *ka.add(10) = k as u32;
    *ka.add(11) = n as u32;
    *ka.add(12) = gs as u32;
    *ka.add(13) = 0;
    *ka.add(14) = 0;

    let mut ib = IbBuilder::new();
    ib.state(&reg);
    ib.dispatch(dd.bos[kabo].mc, (n / MLX4_MV_NUM_ROWS as usize) as u32);
    let words = ib.finish();
    dd.bos[cmdbo].write_u32(&words);
    dd.submit_wait(cmdbo, words.len() as u32, &[kbo, wbo, sbo, bbo, xbo, obo, kabo, cmdbo])?;

    let got = dd.bos[obo].read_f32(n);
    let (want, refsrc): (Vec<f64>, &str) = if let Some(vk) = &vk {
        (vk.iter().map(|&v| v as f64).collect(), "LIVE Vulkan capture")
    } else {
        let mut refv = vec![0f64; n];
        for r in 0..n {
            let mut acc = 0f64;
            for j in 0..k {
                let g = j / gs;
                let word = packed[r * wpr + j / 8];
                let q = ((word >> ((j % 8) * 4)) & 0xF) as f32;
                let w = scales[r * groups + g] * q + biases[r * groups + g];
                acc += w as f64 * x[j] as f64;
            }
            refv[r] = acc;
        }
        (refv, "double CPU ref")
    };
    let (cos, amg, amw, maxe) = cos_argmax(&got, &want);
    let thresh = if vk.is_some() { 1e-3 } else { 1.0 };
    let ok = cos >= 0.99999 && amg == amw && maxe < thresh;
    Ok(format!(
        "mlx4 4-bit matvec  k={k} n={n} (vs {refsrc}): cos={cos:.10} argmax {amg}=={amw} max_abs_err={maxe:.3e} -> {}",
        if ok { "PASS" } else { "*** FAIL ***" }
    ))
}

/// REPLAY 3 — 2-dispatch dependent chain in ONE IB with the inter-dispatch barrier:
/// y = A·x  --[barrier]-->  z = B·y. Proves the multi-dispatch IB builder + barrier
/// + L2 coherence between dependent same-queue dispatches. Gate vs double CPU ref.
unsafe fn replay_chain(dd: &mut DirectDevice, k: usize, mid: usize, out: usize) -> Result<String, String> {
    let mut rng = Rng(0xBEEF_1234_5678);
    let x: Vec<f32> = (0..k).map(|_| rng.f()).collect();
    let a: Vec<f32> = (0..mid * k).map(|_| rng.f()).collect(); // [mid, k]
    let b: Vec<f32> = (0..out * mid).map(|_| rng.f()).collect(); // [out, mid]

    let reg = dd.register_kernel(&MATVEC_F32)?;
    let kbo = reg_bo(dd);
    let abo = dd.gtt((a.len() * 4) as u64)?;
    let bbo = dd.gtt((b.len() * 4) as u64)?;
    let xbo = dd.gtt((k * 4) as u64)?;
    let ybo = dd.gtt((mid * 4) as u64)?; // intermediate (produced then consumed)
    let zbo = dd.gtt((out * 4) as u64)?;
    let ka1 = dd.gtt(4096)?;
    let ka2 = dd.gtt(4096)?;
    let cmdbo = dd.gtt(4096)?;
    dd.bos[abo].write_f32(&a);
    dd.bos[bbo].write_f32(&b);
    dd.bos[xbo].write_f32(&x);
    // dispatch 1: y = A·x  (W=A, x=x, out=y, K=k)
    let p1 = dd.bos[ka1].as_u32();
    *(p1 as *mut u64).add(0) = dd.bos[abo].mc;
    *(p1 as *mut u64).add(1) = dd.bos[xbo].mc;
    *(p1 as *mut u64).add(2) = dd.bos[ybo].mc;
    *p1.add(6) = k as u32;
    // dispatch 2: z = B·y  (W=B, x=y, out=z, K=mid)
    let p2 = dd.bos[ka2].as_u32();
    *(p2 as *mut u64).add(0) = dd.bos[bbo].mc;
    *(p2 as *mut u64).add(1) = dd.bos[ybo].mc;
    *(p2 as *mut u64).add(2) = dd.bos[zbo].mc;
    *p2.add(6) = mid as u32;

    let mut ib = IbBuilder::new();
    ib.state(&reg);
    ib.dispatch(dd.bos[ka1].mc, (mid / 64) as u32);
    ib.barrier();
    ib.state(&reg); // same code object, re-emit state after the barrier
    ib.dispatch(dd.bos[ka2].mc, (out / 64) as u32);
    let words = ib.finish();
    dd.bos[cmdbo].write_u32(&words);
    dd.submit_wait(
        cmdbo,
        words.len() as u32,
        &[kbo, abo, bbo, xbo, ybo, zbo, ka1, ka2, cmdbo],
    )?;

    let got = dd.bos[zbo].read_f32(out);
    // CPU ref: y = A·x ; z = B·y  (double)
    let mut y = vec![0f64; mid];
    for r in 0..mid {
        let mut acc = 0f64;
        for c in 0..k {
            acc += a[r * k + c] as f64 * x[c] as f64;
        }
        y[r] = acc;
    }
    let mut refv = vec![0f64; out];
    for r in 0..out {
        let mut acc = 0f64;
        for c in 0..mid {
            acc += b[r * mid + c] as f64 * y[c];
        }
        refv[r] = acc;
    }
    let (cos, amg, amw, maxe) = cos_argmax(&got, &refv);
    let ok = cos >= 0.99999 && amg == amw;
    Ok(format!(
        "barriered chain  A[{mid},{k}]->[barrier]->B[{out},{mid}]: cos={cos:.10} argmax {amg}=={amw} max_abs_err={maxe:.3e} -> {}",
        if ok { "PASS" } else { "*** FAIL ***" }
    ))
}

// ── Python entry point (M0 gate harness) ────────────────────────────────────────
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// M0 gate: replay the S3 f32 matvec, the s4mlx4 4-bit matvec, and a 2-dispatch
/// barriered chain through the engine's `DirectDevice`, each bit-exact vs a
/// double-precision CPU reference. If `mlx4_cap` is a directory holding a live
/// Vulkan capture ({packed,scales,biases,x,vk_out}.bin), the mlx4 replay is ALSO
/// gated byte-for-byte against that capture. Returns (all_pass, [per-replay lines]).
#[pyfunction]
#[pyo3(signature = (render_node = "/dev/dri/renderD128", mlx4_cap = None))]
pub(crate) fn debug_direct_engine_replay(
    render_node: &str,
    mlx4_cap: Option<&str>,
) -> PyResult<(bool, Vec<String>)> {
    let flag = DirectDevice::flag_enabled();
    let mut lines = vec![format!(
        "DirectEngine M0 replay  (VLLM_VULKAN_DIRECT_SUBMIT={})",
        if flag { "1" } else { "unset (debug replay runs regardless)" }
    )];

    let run = || -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        let mut dd = DirectDevice::open(render_node)?;
        unsafe {
            out.push(replay_s3_matvec(&mut dd, 4352, 5120)?);
            out.push(replay_mlx4(&mut dd, 4352, 5120, None)?);
            if let Some(dir) = mlx4_cap {
                // Derive k,n from the capture file sizes (x.bin=k f32, vk_out.bin=n f32).
                let k = std::fs::metadata(format!("{dir}/x.bin"))
                    .map_err(|e| format!("{dir}/x.bin: {e}"))?
                    .len() as usize
                    / 4;
                let n = std::fs::metadata(format!("{dir}/vk_out.bin"))
                    .map_err(|e| format!("{dir}/vk_out.bin: {e}"))?
                    .len() as usize
                    / 4;
                out.push(replay_mlx4(&mut dd, k, n, Some(dir))?);
            }
            out.push(replay_chain(&mut dd, 4352, 5120, 5120)?);
        }
        Ok(out)
    };

    match run() {
        Ok(res) => {
            let all_pass = res.iter().all(|l| l.ends_with("PASS"));
            lines.extend(res);
            Ok((all_pass, lines))
        }
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "DirectEngine replay failed: {e}"
        ))),
    }
}
