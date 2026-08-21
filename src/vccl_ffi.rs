//! Native Rust FFI binding to libvccl (Vulkan Collective Communications Library).
//!
//! Replaces the per-call Python `PyObject` collective callback (Rust→Python→vCCL
//! →Python→Rust with element-wise `Vec<f32>`↔PyList marshalling + GIL, ~7 ms/call
//! for a 20 KB reduce) with a direct C call on the partial's host memory. The
//! `vcclComm_t` handle is created and owned by the Python launcher (it keeps the
//! bootstrap recipe); the launcher passes the raw pointer into Rust ONCE via
//! `VulkanModel::set_collective_comm` before the forward loop.
//!
//! Library resolution mirrors how `ash` dlopens libvulkan: we `dlopen`
//! `libvccl.so` at runtime (no link-time dependency, so the Mac cross-build needs
//! no libvccl), resolving the four symbols we use. `libvccl.so` is already
//! deployed on every node (the vccl python-binding ships next to it).
//!
//! vCCL is **always host-blocking** (per `vccl.h`): no stream/sync handling. We
//! drop the GIL around the blocking call so the runtime stays responsive and
//! peers can make progress.

use libloading::Library;
use std::os::raw::{c_int, c_void};
use std::sync::OnceLock;

/// `vcclDataType_t` (subset we use).
pub const VCCL_FLOAT32: c_int = 7;
/// `vcclDataType_t::vcclBfloat16` (from include/vccl.h). Half the wire bytes of
/// f32; vCCL reduces in bf16 (the microbench's `DTYPE=bf16` path). The residual
/// stream is summed in bf16 precision when this datatype is used.
pub const VCCL_BFLOAT16: c_int = 9;
/// `vcclRedOp_t::vcclSum`.
pub const VCCL_SUM: c_int = 0;

/// Per-reduce phase breakdown (nanoseconds), so the caller can attribute the
/// in-model ~6.3 ms/reduce to copy-in (GPU-out→registered scratch), wire (the
/// blocking `vcclAllReduce` itself = reduce-scatter/recdouble over RoCE), and
/// copy-out (scratch→partial). STEP-0 of the comm-overlap work: find where the
/// ~5.5 ms above the isolated wire floor lives before picking a lever.
#[derive(Clone, Copy, Default)]
pub struct Phases {
    pub copy_in_ns: u128,
    pub wire_ns: u128,
    pub copy_out_ns: u128,
}

// C signatures (from include/vccl.h). `vcclComm_t` is an opaque pointer.
//   vcclResult_t vcclAllReduce(const void* sendbuff, void* recvbuff,
//                              size_t count, vcclDataType_t datatype,
//                              vcclRedOp_t op, vcclComm_t comm);
//   vcclResult_t vcclSend(const void* sendbuff, size_t count,
//                         vcclDataType_t datatype, int peer, vcclComm_t comm);
//   vcclResult_t vcclRecv(void* recvbuff, size_t count,
//                         vcclDataType_t datatype, int peer, vcclComm_t comm);
//   const char* vcclGetErrorString(vcclResult_t result);
type AllReduceFn = unsafe extern "C" fn(
    *const c_void, *mut c_void, usize, c_int, c_int, *mut c_void,
) -> c_int;
type SendFn =
    unsafe extern "C" fn(*const c_void, usize, c_int, c_int, *mut c_void) -> c_int;
type RecvFn =
    unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, *mut c_void) -> c_int;
//   vcclResult_t vcclSendRecv(const void* sbuf, size_t scount, int sendPeer,
//                             void* rbuf, size_t rcount, int recvPeer,
//                             vcclDataType_t datatype, vcclComm_t comm);
// Duplex exchange primitive (comm-floor Lever 3): posts the recv, posts the
// send, polls BOTH completions off one CQ (full duplex, RNR-optimal) when
// `VCCL_DUPLEX_OVERLAP=1`; unset → deadlock-free ordered half-duplex fallback
// (lower rank sends first, higher recvs first — byte-identical to today's TP=2
// reduce), so ONE binary A/Bs duplex-vs-serial. sbuf/rbuf MUST be distinct.
// Not present in older libvccl builds → dlsym'd `Option<..>` (same best-effort
// posture as all_gather); an absent symbol degrades to the ordered send/recv
// pair in `nem_tp_reduce_mix`, never fails the load.
type SendRecvFn = unsafe extern "C" fn(
    *const c_void, usize, c_int,
    *mut c_void, usize, c_int,
    c_int, *mut c_void,
) -> c_int;
type ErrStrFn = unsafe extern "C" fn(c_int) -> *const std::os::raw::c_char;
//   vcclResult_t vcclCommRegister(vcclComm_t comm, void* buff, size_t bytes,
//                                 vcclMemHandle_t* handle);
//   vcclResult_t vcclCommDeregister(vcclComm_t comm, vcclMemHandle_t handle);
// Pre-registering a buffer pins it with the RDMA transport ONCE; the per-call
// ScopedReg in each collective then short-circuits (covers()) instead of
// ibv_reg_mr/dereg every call — the dominant cost for small reduces.
type CommRegisterFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *mut *mut c_void) -> c_int;
type CommDeregisterFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
//   vcclResult_t vcclAllGather(const void* sendbuff, void* recvbuff,
//                              size_t sendcount, vcclDataType_t datatype,
//                              vcclComm_t comm);
// Uniform sendcount per rank; recvbuff is `nranks * sendcount` elements,
// ordered by rank (recv[r*sendcount .. (r+1)*sendcount] = rank r's send). Not
// present in older libvccl builds — dlsym'd as `Option<..>` (mirrors
// comm_register/deregister) so an old library degrades to replicated lm_head
// with a warning instead of failing the whole load (R3 in the plan).
type AllGatherFn =
    unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, *mut c_void) -> c_int;
//   vcclResult_t vcclAllToAll(const void* sendbuff, void* recvbuff,
//                             size_t count, vcclDataType_t datatype,
//                             vcclComm_t comm);
// RCCL extension (vccl.h): every rank sends `count` elements to every other
// rank; sendbuff/recvbuff hold `nranks*count` elements, block `r` for rank
// `r`. Uniform per-rank count (the balanced-EP case); the ragged case is
// vcclAllToAllv below. EP greenfield (plan-tp-ep-levers.md §6b build-list
// item 3) — no forward path calls this yet, so it is dlsym'd `Option<..>`
// exactly like `vcclAllGather` and is unexercised beyond the no-lib probe
// test until an EP forward exists.
type AllToAllFn =
    unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, *mut c_void) -> c_int;
//   vcclResult_t vcclAllToAllv(const void* sendbuff, const size_t sendcounts[],
//                              const size_t sdispls[], void* recvbuff,
//                              const size_t recvcounts[], const size_t rdispls[],
//                              vcclDataType_t datatype, vcclComm_t comm);
// RCCL extension (vccl.h): variable/ragged all-to-all. counts/displacements
// are in ELEMENTS; the block sent to (received from) rank r starts at
// sdispls[r] (rdispls[r]); rank r's sendcounts[r] must equal rank r's peer's
// recvcounts[rank]. This is the EP expert-scatter/gather primitive (§2.1's
// per-layer "scatter selected experts / gather partials" round trip) — the
// binding EP needs once a batched forward exists. Same Option-dlsym pattern:
// absent in libvccl builds without the RCCL-extension collectives (the vccl
// C side already ships them per collectives.cc:865, but we still probe
// rather than assume, mirroring every other optional symbol in this file).
type AllToAllvFn = unsafe extern "C" fn(
    *const c_void,
    *const usize,
    *const usize,
    *mut c_void,
    *const usize,
    *const usize,
    c_int,
    *mut c_void,
) -> c_int;

/// Resolved libvccl entry points. Loaded once, lives for the process.
struct VcclApi {
    _lib: Library, // keep the library mapped for the life of the process
    all_reduce: AllReduceFn,
    send: SendFn,
    recv: RecvFn,
    send_recv: Option<SendRecvFn>,
    err_str: ErrStrFn,
    comm_register: Option<CommRegisterFn>,
    comm_deregister: Option<CommDeregisterFn>,
    all_gather: Option<AllGatherFn>,
    all_to_all: Option<AllToAllFn>,
    all_to_all_v: Option<AllToAllvFn>,
}

// The function pointers are into a permanently-mapped library; the handle and
// the comm pointer are used under the GIL-released blocking call. vCCL itself is
// single-threaded per comm from our usage (one forward at a time).
unsafe impl Send for VcclApi {}
unsafe impl Sync for VcclApi {}

static API: OnceLock<VcclApi> = OnceLock::new();

/// Process-global reentrancy guard held for the full duration of every native
/// collective (including the GIL-released wire call — this is a plain Rust
/// mutex, independent of the GIL). The Python-owned `vcclComm_t` handle has no
/// Rust-side lifetime tie, so without this a second Python thread could
/// re-enter a collective on the same comm (libvccl data race) or tear the comm
/// down mid-flight (UAF). Collectives are already serial on the single forward
/// thread in normal use, so this is zero-cost insurance. Leaf lock: never held
/// while acquiring another lock.
static COMM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Expose the comm lock so callers can serialize comm invalidation
/// (e.g. zeroing the stored handle) against an in-flight collective.
pub fn comm_lock() -> &'static std::sync::Mutex<()> {
    &COMM_LOCK
}

/// Candidate library names/paths, mirroring the python binding's search order.
fn candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("VCCL_LIBRARY") {
        if !p.is_empty() {
            v.push(p);
        }
    }
    // The deployed library is the soname `libvccl.so.0` (no unversioned dev
    // symlink on the appliance), so try that too. dlopen resolves these via the
    // loader's search path (ldconfig: /lib64/libvccl.so.0).
    v.push("libvccl.so.0".to_string());
    v.push("/lib64/libvccl.so.0".to_string());
    v.push("libvccl.so".to_string());
    v.push("libvccl.dylib".to_string());
    v
}

fn load_api() -> Result<VcclApi, String> {
    let mut last_err = String::new();
    for cand in candidates() {
        let lib = unsafe { Library::new(&cand) };
        let lib = match lib {
            Ok(l) => l,
            Err(e) => {
                last_err = format!("{cand}: {e}");
                continue;
            }
        };
        // Resolve symbols; bail (try next candidate) if any are missing.
        let api = unsafe {
            let all_reduce = lib
                .get::<AllReduceFn>(b"vcclAllReduce\0")
                .map(|s| *s)
                .map_err(|e| format!("{cand}: vcclAllReduce: {e}"));
            let send = lib
                .get::<SendFn>(b"vcclSend\0")
                .map(|s| *s)
                .map_err(|e| format!("{cand}: vcclSend: {e}"));
            let recv = lib
                .get::<RecvFn>(b"vcclRecv\0")
                .map(|s| *s)
                .map_err(|e| format!("{cand}: vcclRecv: {e}"));
            // Optional (comm-floor Lever 3): the duplex exchange primitive.
            // Absent in older libvccl builds → degrade to the ordered send/recv
            // pair in `nem_tp_reduce_mix`, same best-effort posture as
            // all_gather. Present in libvccl.so.0.2.0 (md5 6a3d6958).
            let send_recv = lib
                .get::<SendRecvFn>(b"vcclSendRecv\0")
                .map(|s| *s)
                .ok();
            let err_str = lib
                .get::<ErrStrFn>(b"vcclGetErrorString\0")
                .map(|s| *s)
                .map_err(|e| format!("{cand}: vcclGetErrorString: {e}"));
            // Registration is optional (older libvccl may lack it): resolve
            // best-effort, fall back to per-call ScopedReg when absent.
            let comm_register = lib
                .get::<CommRegisterFn>(b"vcclCommRegister\0")
                .map(|s| *s)
                .ok();
            let comm_deregister = lib
                .get::<CommDeregisterFn>(b"vcclCommDeregister\0")
                .map(|s| *s)
                .ok();
            // Optional: absent in older libvccl builds (R3 — degrade to
            // replicated lm_head with a warning, never fail the whole load).
            let all_gather = lib
                .get::<AllGatherFn>(b"vcclAllGather\0")
                .map(|s| *s)
                .ok();
            // Optional (EP greenfield): the RCCL-extension all-to-all(v)
            // collectives. No caller today, so an absent symbol degrades to
            // "EP unavailable" rather than failing the load — the same
            // best-effort posture as `all_gather`.
            let all_to_all = lib
                .get::<AllToAllFn>(b"vcclAllToAll\0")
                .map(|s| *s)
                .ok();
            let all_to_all_v = lib
                .get::<AllToAllvFn>(b"vcclAllToAllv\0")
                .map(|s| *s)
                .ok();
            match (all_reduce, send, recv, err_str) {
                (Ok(a), Ok(s), Ok(r), Ok(e)) => VcclApi {
                    _lib: lib,
                    all_reduce: a,
                    send: s,
                    recv: r,
                    send_recv,
                    err_str: e,
                    comm_register,
                    comm_deregister,
                    all_gather,
                    all_to_all,
                    all_to_all_v,
                },
                (a, s, r, e) => {
                    last_err = [a.err(), s.err(), r.err(), e.err()]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join("; ");
                    continue;
                }
            }
        };
        return Ok(api);
    }
    Err(format!(
        "cannot load libvccl (set VCCL_LIBRARY); last error: {last_err}"
    ))
}

fn api() -> Result<&'static VcclApi, String> {
    // Cache only SUCCESS. A failed load (e.g. VCCL_LIBRARY not yet set) must stay
    // retryable — OnceLock<Result<..>> would pin the failure for the process life.
    if let Some(a) = API.get() {
        return Ok(a);
    }
    let loaded = load_api()?; // propagate the error WITHOUT caching it
    Ok(API.get_or_init(|| loaded))
}

/// True if libvccl could be dlopened + all symbols resolved.
pub fn available() -> bool {
    api().is_ok()
}

fn err_string(api: &VcclApi, code: c_int) -> String {
    if code == 0 {
        return "success".to_string();
    }
    unsafe {
        let p = (api.err_str)(code);
        if p.is_null() {
            format!("vccl error {code}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// In-place all-reduce(SUM, FLOAT32) over `buf` across the communicator. `comm`
/// is the raw `vcclComm_t` the launcher handed us. vCCL supports in-place when
/// `sendbuff == recvbuff`, so this is a pure reduce on the partial's own memory:
/// no allocation, no marshalling, no Python. Releases the GIL around the
/// host-blocking call so peers progress and the interpreter stays responsive.
pub fn all_reduce_f32_sum_inplace(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    buf: &mut [f32],
) -> Result<Phases, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null (set_collective_comm not called?)".into());
    }
    let n = buf.len();
    // Move raw addresses as usize across the allow_threads boundary (raw
    // pointers aren't Send). vCCL is host-blocking; we don't touch `buf` until
    // the call returns, so dropping the GIL here is sound.
    let ptr_u = buf.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    let f = api.all_reduce;
    let t_w = std::time::Instant::now();
    let code = py.allow_threads(move || unsafe {
        let ptr = ptr_u as *mut c_void;
        f(ptr as *const c_void, ptr, n, VCCL_FLOAT32, VCCL_SUM, comm_u as *mut c_void)
    });
    let wire_ns = t_w.elapsed().as_nanos();
    if code != 0 {
        return Err(format!("vcclAllReduce: {}", err_string(api, code)));
    }
    Ok(Phases { copy_in_ns: 0, wire_ns, copy_out_ns: 0 })
}

/// In-place all-reduce(SUM, **BFLOAT16**) over `buf`. Converts the f32 partial →
/// bf16 (copy_in), reduces with `VCCL_BFLOAT16` over the wire (half the f32
/// bytes, and vCCL sums in bf16), then converts the bf16 result → f32 (copy_out)
/// back into `buf`. The precision cost is real: each rank's partial is truncated
/// to bf16 (8-bit mantissa) before the sum, and the running reduce accumulates in
/// bf16 — so the residual stream loses ~16 mantissa bits per reduce vs the f32
/// path. Caller must measure whether that flips argmax / tanks cos.
///
/// Uses a fresh `Vec<u16>` for the bf16 staging (NOT registered with vCCL → pays
/// per-call ibv_reg_mr; the bf16 win is the halved wire payload, and the
/// scratch-registered bf16 path is `all_reduce_bf16_via_scratch`).
pub fn all_reduce_bf16_sum_inplace(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    buf: &mut [f32],
) -> Result<Phases, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null (set_collective_comm not called?)".into());
    }
    let n = buf.len();
    // f32 → bf16 (round-to-nearest-even via the half crate's bf16).
    let t_ci = std::time::Instant::now();
    let mut staging: Vec<u16> = buf
        .iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect();
    let copy_in_ns = t_ci.elapsed().as_nanos();
    let ptr_u = staging.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    let f = api.all_reduce;
    let t_w = std::time::Instant::now();
    let code = py.allow_threads(move || unsafe {
        let ptr = ptr_u as *mut c_void;
        f(ptr as *const c_void, ptr, n, VCCL_BFLOAT16, VCCL_SUM, comm_u as *mut c_void)
    });
    let wire_ns = t_w.elapsed().as_nanos();
    if code != 0 {
        return Err(format!("vcclAllReduce(bf16): {}", err_string(api, code)));
    }
    // bf16 → f32 back into the caller's buffer.
    let t_co = std::time::Instant::now();
    for (dst, &src) in buf.iter_mut().zip(staging.iter()) {
        *dst = half::bf16::from_bits(src).to_f32();
    }
    let copy_out_ns = t_co.elapsed().as_nanos();
    Ok(Phases { copy_in_ns, wire_ns, copy_out_ns })
}

/// bf16 all-reduce(SUM) of `partial` THROUGH a pre-registered scratch (capacity
/// `scratch_len` **f32 slots** = `2*scratch_len` bytes; bf16 needs 2 bytes/elt so
/// `n` bf16 elements fit in `scratch_len` f32 slots whenever `n <= 2*scratch_len`,
/// but we conservatively require `n <= scratch_len` to reuse the same registered
/// region). The bf16 staging lives in the registered scratch so vCCL's per-call
/// ScopedReg short-circuits (no ibv_reg_mr). Convert in, reduce, convert out.
///
/// # Safety
/// `scratch_addr`..+(`2*n` bytes) must be a valid, uniquely-owned region
/// registered with `comm`, not aliasing `partial`.
pub unsafe fn all_reduce_bf16_via_scratch(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    scratch_addr: usize,
    scratch_len: usize,
    partial: &mut [f32],
) -> Result<Phases, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let n = partial.len();
    // scratch is sized in f32 slots (4 bytes); bf16 uses 2 bytes/elt, so n bf16
    // elements need n*2 bytes <= scratch_len*4 bytes → n <= 2*scratch_len.
    if n > scratch_len * 2 {
        return Err(format!(
            "bf16 reduce len {n} exceeds registered scratch {} bf16 slots",
            scratch_len * 2
        ));
    }
    if scratch_addr == 0 {
        return Err("scratch_addr is null".into());
    }
    if scratch_addr % std::mem::align_of::<u16>() != 0 {
        return Err(format!("scratch_addr {scratch_addr:#x} not u16-aligned"));
    }
    let scratch = std::slice::from_raw_parts_mut(scratch_addr as *mut u16, n);
    let t_ci = std::time::Instant::now();
    for (dst, &src) in scratch.iter_mut().zip(partial.iter()) {
        *dst = half::bf16::from_f32(src).to_bits();
    }
    let copy_in_ns = t_ci.elapsed().as_nanos();
    let ptr_u = scratch_addr;
    let comm_u = comm as usize;
    let f = api.all_reduce;
    let t_w = std::time::Instant::now();
    let code = py.allow_threads(move || {
        let ptr = ptr_u as *mut c_void;
        f(ptr as *const c_void, ptr, n, VCCL_BFLOAT16, VCCL_SUM, comm_u as *mut c_void)
    });
    let wire_ns = t_w.elapsed().as_nanos();
    if code != 0 {
        return Err(format!("vcclAllReduce(bf16,scratch): {}", err_string(api, code)));
    }
    let t_co = std::time::Instant::now();
    for (dst, &src) in partial.iter_mut().zip(scratch.iter()) {
        *dst = half::bf16::from_bits(src).to_f32();
    }
    let copy_out_ns = t_co.elapsed().as_nanos();
    Ok(Phases { copy_in_ns, wire_ns, copy_out_ns })
}

/// All-reduce(SUM, FLOAT32) of `partial` THROUGH a pre-registered scratch
/// buffer at `scratch_addr` (capacity `scratch_len` floats). vCCL's per-call
/// `ScopedReg` short-circuits because the scratch is already covered by a prior
/// `vcclCommRegister`, so no `ibv_reg_mr`/dereg per reduce. The partial is
/// copied in, reduced in place, copied back. `scratch_addr` must point to the
/// registered buffer and be large enough; caller guarantees it does not alias
/// `partial`. GIL released around the blocking reduce.
///
/// # Safety
/// `scratch_addr`..+`scratch_len` must be a valid, uniquely-owned `f32` region
/// registered with `comm`, not aliasing `partial`.
pub unsafe fn all_reduce_via_scratch(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    scratch_addr: usize,
    scratch_len: usize,
    partial: &mut [f32],
) -> Result<Phases, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let n = partial.len();
    if n > scratch_len {
        return Err(format!("reduce len {n} exceeds registered scratch {scratch_len}"));
    }
    if scratch_addr == 0 {
        return Err("scratch_addr is null".into());
    }
    if scratch_addr % std::mem::align_of::<f32>() != 0 {
        return Err(format!("scratch_addr {scratch_addr:#x} not f32-aligned"));
    }
    let scratch = std::slice::from_raw_parts_mut(scratch_addr as *mut f32, n);
    let t_ci = std::time::Instant::now();
    scratch.copy_from_slice(partial);
    let copy_in_ns = t_ci.elapsed().as_nanos();
    let ptr_u = scratch_addr;
    let comm_u = comm as usize;
    let f = api.all_reduce;
    let t_w = std::time::Instant::now();
    let code = py.allow_threads(move || {
        let ptr = ptr_u as *mut c_void;
        f(ptr as *const c_void, ptr, n, VCCL_FLOAT32, VCCL_SUM, comm_u as *mut c_void)
    });
    let wire_ns = t_w.elapsed().as_nanos();
    if code != 0 {
        return Err(format!("vcclAllReduce(scratch): {}", err_string(api, code)));
    }
    let t_co = std::time::Instant::now();
    partial.copy_from_slice(scratch);
    let copy_out_ns = t_co.elapsed().as_nanos();
    Ok(Phases { copy_in_ns, wire_ns, copy_out_ns })
}

/// Blocking send of `buf` (FLOAT32) to `peer`. GIL released around the call.
pub fn send_f32(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    buf: &[f32],
    peer: i32,
) -> Result<(), String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let n = buf.len();
    let ptr_u = buf.as_ptr() as usize;
    let comm_u = comm as usize;
    let f = api.send;
    let code = py.allow_threads(move || unsafe {
        f(ptr_u as *const c_void, n, VCCL_FLOAT32, peer as c_int, comm_u as *mut c_void)
    });
    if code != 0 {
        return Err(format!("vcclSend: {}", err_string(api, code)));
    }
    Ok(())
}

/// Blocking recv of `n` FLOAT32 elements from `peer` into a fresh buffer.
pub fn recv_f32(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    n: usize,
    peer: i32,
) -> Result<Vec<f32>, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let mut buf = vec![0.0f32; n];
    let ptr_u = buf.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    let f = api.recv;
    let code = py.allow_threads(move || unsafe {
        f(ptr_u as *mut c_void, n, VCCL_FLOAT32, peer as c_int, comm_u as *mut c_void)
    });
    if code != 0 {
        return Err(format!("vcclRecv: {}", err_string(api, code)));
    }
    Ok(buf)
}

/// Blocking recv of `dst.len()` FLOAT32 elements from `peer` IN PLACE into
/// `dst` — no fresh `Vec` alloc (unlike [`recv_f32`]). For the pre-registered
/// TP-reduce scratch (`nemotron.rs` lever 1): when `dst` was `comm_register`'d
/// once, this also skips the per-call `ibv_reg_mr`/dereg. GIL released around
/// the blocking wire call.
pub fn recv_f32_into(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    dst: &mut [f32],
    peer: i32,
) -> Result<(), String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let n = dst.len();
    let ptr_u = dst.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    let f = api.recv;
    let code = py.allow_threads(move || unsafe {
        f(ptr_u as *mut c_void, n, VCCL_FLOAT32, peer as c_int, comm_u as *mut c_void)
    });
    if code != 0 {
        return Err(format!("vcclRecv: {}", err_string(api, code)));
    }
    Ok(())
}

/// True if libvccl exposes the duplex exchange primitive (`vcclSendRecv`,
/// comm-floor Lever 3). When false, callers fall back to the ordered
/// `send_f32` + `recv_f32_into` pair.
pub fn send_recv_available() -> bool {
    api().map(|a| a.send_recv.is_some()).unwrap_or(false)
}

/// Duplex exchange (FLOAT32): simultaneously send `sbuf` to `send_peer` and
/// recv `rbuf.len()` elements from `recv_peer` in ONE `vcclSendRecv` call.
/// `sbuf`/`rbuf` MUST be distinct buffers (they are in the TP reduce: `partial`
/// vs `theirs`). The library picks full-duplex (post recv, post send, poll both
/// off one CQ) when `VCCL_DUPLEX_OVERLAP=1`, else a deadlock-free ordered
/// half-duplex fallback (lower rank sends first) that is byte-identical to the
/// legacy `send_f32`+`recv_f32_into` pair — so one binary A/Bs both modes. GIL
/// released around the blocking wire call. Only the send/recv WAIT ORDER
/// differs between modes; the bytes exchanged (and thus the reduce result) are
/// identical. Requires [`send_recv_available`]; errors if the symbol is absent.
pub fn send_recv_f32(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    sbuf: &[f32],
    send_peer: i32,
    rbuf: &mut [f32],
    recv_peer: i32,
) -> Result<(), String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    let f = api
        .send_recv
        .ok_or_else(|| "vcclSendRecv unavailable in this libvccl".to_string())?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let scount = sbuf.len();
    let rcount = rbuf.len();
    let sptr_u = sbuf.as_ptr() as usize;
    let rptr_u = rbuf.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    let code = py.allow_threads(move || unsafe {
        f(
            sptr_u as *const c_void,
            scount,
            send_peer as c_int,
            rptr_u as *mut c_void,
            rcount,
            recv_peer as c_int,
            VCCL_FLOAT32,
            comm_u as *mut c_void,
        )
    });
    if code != 0 {
        return Err(format!("vcclSendRecv: {}", err_string(api, code)));
    }
    Ok(())
}

/// True if libvccl exposes the pre-registration entry points.
pub fn registration_available() -> bool {
    api()
        .map(|a| a.comm_register.is_some() && a.comm_deregister.is_some())
        .unwrap_or(false)
}

/// Pre-register `bytes` at `addr` with the communicator's RDMA transport, so
/// subsequent collectives on that buffer skip the per-call `ibv_reg_mr`/dereg
/// (the dominant cost for small reduces — see vccl ScopedReg::add covers()).
/// Returns the opaque handle (as usize) to pass back to `comm_deregister`.
pub fn comm_register(comm: *mut c_void, addr: usize, bytes: usize) -> Result<usize, String> {
    let api = api()?;
    let f = api
        .comm_register
        .ok_or_else(|| "vcclCommRegister unavailable in this libvccl".to_string())?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let mut handle: *mut c_void = std::ptr::null_mut();
    let code = unsafe { f(comm, addr as *mut c_void, bytes, &mut handle) };
    if code != 0 {
        return Err(format!("vcclCommRegister: {}", err_string(api, code)));
    }
    Ok(handle as usize)
}

/// Deregister a handle previously returned by `comm_register`.
pub fn comm_deregister(comm: *mut c_void, handle: usize) -> Result<(), String> {
    let api = api()?;
    let f = api
        .comm_deregister
        .ok_or_else(|| "vcclCommDeregister unavailable".to_string())?;
    if comm.is_null() || handle == 0 {
        return Ok(());
    }
    let code = unsafe { f(comm, handle as *mut c_void) };
    if code != 0 {
        return Err(format!("vcclCommDeregister: {}", err_string(api, code)));
    }
    Ok(())
}

/// True if the loaded libvccl exposes `vcclAllGather` (absent in some older
/// builds — R3: callers must fall back to replicated lm_head when this is
/// false, never fail the load).
pub fn allgather_available() -> bool {
    api().map(|a| a.all_gather.is_some()).unwrap_or(false)
}

/// All-gather `send` (FLOAT32, uniform `sendcount = send.len()` per rank)
/// across the communicator: returns `nranks * send.len()` floats, rank-ordered
/// (`recv[r*sendcount..(r+1)*sendcount]` = rank r's `send`). Used for the TP
/// vocab-sharded lm_head tail (argmax pack, full-logits, top-k) — see
/// `tp_argmax_merge`/`tp_topk_merge` in `tp.rs` for the merge side. Allocates a
/// fresh recv `Vec` (not RDMA-pre-registered) — for the hot per-token argmax/
/// topk gathers (tiny payload) prefer `all_gather_via_scratch`. GIL released
/// around the blocking wire call, mirroring `all_reduce_f32_sum_inplace`.
pub fn all_gather_f32(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    send: &[f32],
    nranks: usize,
) -> Result<Vec<f32>, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null (set_collective_comm not called?)".into());
    }
    let f = api
        .all_gather
        .ok_or_else(|| "vcclAllGather unavailable in this libvccl".to_string())?;
    let sendcount = send.len();
    let mut recv = vec![0.0f32; sendcount * nranks];
    let send_u = send.as_ptr() as usize;
    let recv_u = recv.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    let t_w = std::time::Instant::now();
    let code = py.allow_threads(move || unsafe {
        f(send_u as *const c_void, recv_u as *mut c_void, sendcount, VCCL_FLOAT32, comm_u as *mut c_void)
    });
    let _wire_ns = t_w.elapsed().as_nanos();
    if code != 0 {
        return Err(format!("vcclAllGather: {}", err_string(api, code)));
    }
    Ok(recv)
}

/// All-gather through a pre-registered pinned scratch buffer (mirrors
/// `all_reduce_via_scratch`): copy `send` into the registered scratch, gather
/// in place (send and recv aliasing the same registered region is NOT what we
/// do here — vCCL's allgather sendbuff/recvbuff are distinct regions of one
/// buffer, so we lay out `scratch[0..sendcount]` = send side and
/// `scratch[sendcount..sendcount+sendcount*nranks]` = recv side within the
/// same registered allocation), then copy the assembled `nranks*sendcount`
/// floats out. Skips the per-call `ibv_reg_mr` that a fresh `Vec` recv buffer
/// would otherwise pay every token on the hot argmax/topk gather path.
///
/// # Safety
/// `scratch_addr..+scratch_len` (in f32 elements) must be a valid, uniquely
/// owned region registered with `comm`, sized at least
/// `send.len() * (1 + nranks)`.
pub unsafe fn all_gather_via_scratch(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    scratch_addr: usize,
    scratch_len: usize,
    send: &[f32],
    nranks: usize,
) -> Result<Vec<f32>, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null".into());
    }
    let f = api
        .all_gather
        .ok_or_else(|| "vcclAllGather unavailable in this libvccl".to_string())?;
    let sendcount = send.len();
    let need = sendcount * (1 + nranks);
    if need > scratch_len {
        return Err(format!("allgather need {need} exceeds registered scratch {scratch_len}"));
    }
    if scratch_addr == 0 {
        return Err("scratch_addr is null".into());
    }
    if scratch_addr % std::mem::align_of::<f32>() != 0 {
        return Err(format!("scratch_addr {scratch_addr:#x} not f32-aligned"));
    }
    let send_scratch = std::slice::from_raw_parts_mut(scratch_addr as *mut f32, sendcount);
    send_scratch.copy_from_slice(send);
    let send_u = scratch_addr;
    let recv_u = scratch_addr + sendcount * std::mem::size_of::<f32>();
    let comm_u = comm as usize;
    let code = py.allow_threads(move || unsafe {
        f(send_u as *const c_void, recv_u as *mut c_void, sendcount, VCCL_FLOAT32, comm_u as *mut c_void)
    });
    if code != 0 {
        return Err(format!("vcclAllGather(scratch): {}", err_string(api, code)));
    }
    let recv_scratch = std::slice::from_raw_parts(recv_u as *const f32, sendcount * nranks);
    Ok(recv_scratch.to_vec())
}

// ── EP all-to-all(v) FFI skeleton ───────────────────────────────────────────
//
// Binding skeleton only (plan-tp-ep-levers.md §6b build-list item 3): wires
// the symbols and a safe wrapper so an EP forward path has something to call
// once one exists (§7 item 7 — greenfield, gated on the batched hybrid
// forward). NOT exercised against a live comm anywhere in the engine yet;
// wire-level validation (a real 2+-rank all-to-all-v round trip) is on the
// deferred list alongside the rest of EP, since the cluster is off-limits for
// this pass. The no-lib Mac test below only proves the dlsym degrades
// gracefully, matching `allgather_available`'s posture for a symbol an old
// (or, here, any current no-EP-caller) libvccl may not export.

/// True if the loaded libvccl exposes `vcclAllToAllv` (the ragged/variable
/// all-to-all — the EP expert-scatter/gather primitive). Mirrors
/// `allgather_available`: false whenever the library can't be loaded at all,
/// or loads but predates/omits the RCCL-extension collectives.
pub fn alltoallv_available() -> bool {
    api().map(|a| a.all_to_all_v.is_some()).unwrap_or(false)
}

/// True if the loaded libvccl exposes the uniform `vcclAllToAll` (every rank
/// sends the same `count` to every other rank — the balanced-EP case, e.g.
/// exactly `experts_per_rank` fixed across ranks). Separate from
/// `alltoallv_available` since a build could plausibly ship one RCCL
/// extension without the other.
pub fn alltoall_available() -> bool {
    api().map(|a| a.all_to_all.is_some()).unwrap_or(false)
}

/// Uniform all-to-all(FLOAT32): every rank sends `count` elements to every
/// other rank. `send` must hold exactly `nranks * count` elements laid out
/// `send[r*count..(r+1)*count]` = the block destined for rank `r`; returns
/// `nranks * count` elements similarly laid out by SOURCE rank (`recv[r*count
/// ..(r+1)*count]` = what rank `r` sent us). Mirrors `all_gather_f32`'s
/// alloc-and-copy posture (no RDMA pre-registration) — the hot EP scatter/
/// gather path should prefer a scratch-backed variant once EP forward exists
/// and the payload sizes are known (same lesson as
/// `all_gather_via_scratch`).
pub fn all_to_all_f32(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    send: &[f32],
    count: usize,
    nranks: usize,
) -> Result<Vec<f32>, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    if comm.is_null() {
        return Err("native comm handle is null (set_collective_comm not called?)".into());
    }
    if send.len() != count * nranks {
        return Err(format!(
            "all_to_all_f32: send.len()={} != count*nranks={}",
            send.len(),
            count * nranks
        ));
    }
    let f = api
        .all_to_all
        .ok_or_else(|| "vcclAllToAll unavailable in this libvccl".to_string())?;
    let mut recv = vec![0.0f32; count * nranks];
    let send_u = send.as_ptr() as usize;
    let recv_u = recv.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    let code = py.allow_threads(move || unsafe {
        f(send_u as *const c_void, recv_u as *mut c_void, count, VCCL_FLOAT32, comm_u as *mut c_void)
    });
    if code != 0 {
        return Err(format!("vcclAllToAll: {}", err_string(api, code)));
    }
    Ok(recv)
}

/// Ragged all-to-all-v(FLOAT32): rank `r` sends `sendcounts[r]` elements
/// (starting at `send[sdispls[r]..]`) to every peer, and receives
/// `recvcounts[r]` elements (placed at `recv[rdispls[r]..]`) from peer `r` —
/// the EP expert-scatter/gather shape, where each rank routes a different
/// number of tokens to each peer depending on the token's top-k expert
/// assignment. `sendcounts`/`sdispls`/`recvcounts`/`rdispls` must each have
/// `nranks` entries (counts/displacements in ELEMENTS, matching
/// `vcclAllToAllv`'s C contract). Caller (the future EP forward) owns
/// computing the ragged counts from the router's expert assignment; this
/// wrapper only owns the FFI call + result buffer sizing.
pub fn all_to_all_v_f32(
    py: pyo3::Python<'_>,
    comm: *mut c_void,
    send: &[f32],
    sendcounts: &[usize],
    sdispls: &[usize],
    recvcounts: &[usize],
    rdispls: &[usize],
    nranks: usize,
) -> Result<Vec<f32>, String> {
    let _serialize = COMM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let api = api()?;
    // Validate shapes before the null-comm check: arity mistakes are a caller
    // bug independent of comm state, and checking first gives a precise error
    // even when comm also happens to be null (as it will be from any test
    // that doesn't have a live comm to hand).
    if sendcounts.len() != nranks || sdispls.len() != nranks
        || recvcounts.len() != nranks || rdispls.len() != nranks
    {
        return Err(format!(
            "all_to_all_v_f32: counts/displs must have nranks={nranks} entries \
             (sendcounts={}, sdispls={}, recvcounts={}, rdispls={})",
            sendcounts.len(), sdispls.len(), recvcounts.len(), rdispls.len()
        ));
    }
    if comm.is_null() {
        return Err("native comm handle is null (set_collective_comm not called?)".into());
    }
    let need_send = sdispls.iter().zip(sendcounts).map(|(&d, &c)| d + c).max().unwrap_or(0);
    if send.len() < need_send {
        return Err(format!(
            "all_to_all_v_f32: send buffer len {} < required {need_send} (sdispls+sendcounts)",
            send.len()
        ));
    }
    let recv_len = rdispls.iter().zip(recvcounts).map(|(&d, &c)| d + c).max().unwrap_or(0);
    let f = api
        .all_to_all_v
        .ok_or_else(|| "vcclAllToAllv unavailable in this libvccl".to_string())?;
    let mut recv = vec![0.0f32; recv_len];
    let send_u = send.as_ptr() as usize;
    let recv_u = recv.as_mut_ptr() as usize;
    let comm_u = comm as usize;
    // sendcounts/sdispls/recvcounts/rdispls must outlive the blocking call;
    // clone into owned Vecs so the `move` closure below can carry them across
    // the allow_threads boundary without borrowing `&[usize]` past its scope.
    let sendcounts = sendcounts.to_vec();
    let sdispls = sdispls.to_vec();
    let recvcounts = recvcounts.to_vec();
    let rdispls = rdispls.to_vec();
    let code = py.allow_threads(move || unsafe {
        f(
            send_u as *const c_void,
            sendcounts.as_ptr(),
            sdispls.as_ptr(),
            recv_u as *mut c_void,
            recvcounts.as_ptr(),
            rdispls.as_ptr(),
            VCCL_FLOAT32,
            comm_u as *mut c_void,
        )
    });
    if code != 0 {
        return Err(format!("vcclAllToAllv: {}", err_string(api, code)));
    }
    Ok(recv)
}

#[cfg(test)]
mod alltoallv_tests {
    use super::*;

    /// On a lib-less Mac (no `libvccl.so`/`libvccl.dylib` on the search
    /// path), `api()` fails to load entirely, so every availability probe —
    /// old (`available`, `allgather_available`) and new
    /// (`alltoall_available`, `alltoallv_available`) — must degrade to
    /// `false` gracefully rather than panicking, mirroring the no-lib
    /// posture the allgather binding already relies on (R3: absent symbol ->
    /// warn/degrade, never fail the whole load).
    #[test]
    fn alltoallv_absent_gracefully_on_libless_mac() {
        // Don't assert `available() == false` unconditionally: some Mac dev
        // boxes may have a stray libvccl.dylib on the search path (e.g. from
        // a prior local vccl build) in which case the *_available() probes
        // legitimately reflect that build's exported symbols instead. What
        // must hold unconditionally is internal consistency: if the library
        // can't load at all, ALL probes agree it's unavailable.
        if !available() {
            assert!(!alltoall_available(), "no lib loaded but alltoall_available() true");
            assert!(!alltoallv_available(), "no lib loaded but alltoallv_available() true");
        }
    }

    /// Calling the wrappers with a null comm must fail cleanly (matching
    /// every other collective wrapper's null-comm guard) instead of
    /// dereferencing — exercised without any live comm/cluster. On a
    /// lib-less Mac `api()` itself fails first (no libvccl to dlopen at
    /// all) — that is also a clean `Err`, so accept either that or the
    /// specific null-comm message; what must NEVER happen is a panic/UB from
    /// dereferencing the null pointer.
    #[test]
    fn alltoall_wrappers_reject_null_comm() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            let send = vec![1.0f32, 2.0, 3.0, 4.0];
            let err = all_to_all_f32(py, std::ptr::null_mut(), &send, 2, 2)
                .expect_err("null comm must error, not dereference");
            assert!(
                err.contains("null") || err.contains("cannot load libvccl"),
                "unexpected error: {err}"
            );

            let counts = vec![2usize, 2];
            let displs = vec![0usize, 2];
            let err = all_to_all_v_f32(
                py, std::ptr::null_mut(), &send, &counts, &displs, &counts, &displs, 2,
            )
            .expect_err("null comm must error, not dereference");
            assert!(
                err.contains("null") || err.contains("cannot load libvccl"),
                "unexpected error: {err}"
            );
        });
    }

    /// Malformed count/displ arity must be rejected before touching FFI. This
    /// check happens before the null-comm check inside `all_to_all_v_f32`,
    /// but ALSO before `api()` is even consulted — wait, it isn't: `api()?`
    /// runs first in the wrapper (needed to report the right error when
    /// libvccl truly can't load), so on a lib-less Mac the load failure wins.
    /// Accept either message: what matters is the arity guard fires on any
    /// build where the library loads (including a real EP validation run).
    #[test]
    fn alltoallv_rejects_mismatched_arity() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            let send = vec![1.0f32, 2.0, 3.0, 4.0];
            let counts = vec![2usize, 2];
            let bad_displs = vec![0usize]; // len 1, nranks=2 -> mismatch
            let err = all_to_all_v_f32(
                py, std::ptr::null_mut(), &send, &counts, &bad_displs, &counts, &bad_displs, 2,
            )
            .expect_err("arity mismatch must error before dereferencing comm");
            assert!(
                err.contains("nranks") || err.contains("cannot load libvccl"),
                "unexpected error: {err}"
            );
        });
    }

    /// Comm-floor Lever 3: on a lib-less Mac the duplex probe must agree with
    /// the global `available()` — if no library loads, `vcclSendRecv` cannot be
    /// resolved, so `send_recv_available()` must be false. Mirrors the
    /// all_gather/all_to_all degrade-gracefully posture (an old libvccl without
    /// the symbol falls back to the ordered send/recv pair in the reduce).
    #[test]
    fn send_recv_absent_gracefully_on_libless_mac() {
        if !available() {
            assert!(
                !send_recv_available(),
                "no lib loaded but send_recv_available() true"
            );
        }
    }

    /// `send_recv_f32` with a null comm (or no library at all) must return a
    /// clean `Err` rather than dereferencing — same null-comm guard as every
    /// other wrapper. `sbuf`/`rbuf` are distinct here, as the TP reduce
    /// requires. Exercised with no live comm/cluster.
    #[test]
    fn send_recv_rejects_null_comm() {
        pyo3::prepare_freethreaded_python();
        pyo3::Python::with_gil(|py| {
            let send = vec![1.0f32, 2.0, 3.0, 4.0];
            let mut recv = vec![0.0f32; 4];
            let err = send_recv_f32(py, std::ptr::null_mut(), &send, 1, &mut recv, 1)
                .expect_err("null comm must error, not dereference");
            // Either the null-comm guard fires (symbol resolved) or the library
            // can't load / lacks vcclSendRecv — all are clean Errs, never UB.
            assert!(
                err.contains("null")
                    || err.contains("cannot load libvccl")
                    || err.contains("vcclSendRecv unavailable"),
                "unexpected error: {err}"
            );
        });
    }
}
