// SPDX-License-Identifier: Apache-2.0
//! Static (no-GPU) SPIR-V reflection gates for the shipped compute shaders.
//!
//! Descriptor-binding and push-constant mismatches between a `.comp` and the
//! Rust code that dispatches it are *silent* on the host: `vkCmdPushConstants`
//! happily writes 20 bytes into a block the shader declared as 24, and
//! `vkUpdateDescriptorSets` happily writes binding 3 for a shader whose third
//! buffer is at binding 4. The failure only shows up on device, as garbage
//! output or a hang — and CI has no GPU.
//!
//! So we reflect the compiled SPIR-V ourselves (a small hand-rolled word-stream
//! parser; no new crate dependency) and assert, statically, the invariants the
//! dispatch path in `compute.rs` / `pipeline.rs` genuinely relies on. See
//! `check_module` for the list, each with its code citation.

use std::collections::HashMap;

// ── SPIR-V constants (SPIR-V 1.6 spec, §3) ─────────────────────────────────

const SPIRV_MAGIC: u32 = 0x0723_0203;

// Opcodes.
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_SPEC_CONSTANT: u16 = 50;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_EXECUTION_MODE_ID: u16 = 331;

// Decorations.
const DEC_BUILT_IN: u32 = 11;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_MATRIX_STRIDE: u32 = 7;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;

const BUILTIN_WORKGROUP_SIZE: u32 = 25;

// Storage classes.
const SC_PUSH_CONSTANT: u32 = 9;

// Execution model / modes.
const EXEC_MODEL_GL_COMPUTE: u32 = 5;
const EXEC_MODE_LOCAL_SIZE: u32 = 17;
const EXEC_MODE_LOCAL_SIZE_ID: u32 = 38;

/// Vulkan's guaranteed-minimum `maxPushConstantsSize`, and exactly the range
/// size every pipeline layout in `pipeline.rs` declares (`PushConstantRange`
/// `.offset(0).size(128)` — see `compile_one_with_spec` and
/// `compile_with_spec_timeout`). A shader whose block exceeds this could not
/// be fed at all.
const PUSH_CONSTANT_RANGE_BYTES: u32 = 128;

// ── Reflection ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    Scalar(u32),          // byte width
    Vector(u32, u32),     // component type id, count
    Matrix(u32, u32),     // column type id, count
    Array(u32, u32),      // element type id, length constant id
    RuntimeArray(u32),    // element type id
    Struct,               // members held in `struct_members`
    Pointer(u32, u32),    // storage class, pointee
}

#[derive(Debug, Default)]
pub struct Reflection {
    /// (major, minor) from the SPIR-V header word 1.
    pub version: (u8, u8),
    /// (execution model, entry-point name).
    pub entry_points: Vec<(u32, String)>,
    /// (descriptor set, binding) for every decorated variable, sorted.
    pub bindings: Vec<(u32, u32)>,
    /// Size in bytes of the push-constant block, if the module declares one.
    pub push_constant_size: Option<u32>,
    /// Literal `LocalSize` execution mode, if present.
    pub local_size: Option<(u32, u32, u32)>,
    /// True when the workgroup size comes from specialization constants
    /// (`local_size_x_id` → `BuiltIn WorkgroupSize`, or `LocalSizeId`).
    pub local_size_is_spec: bool,
}

/// Parse `spv` (a little-endian SPIR-V binary) far enough to answer the
/// questions the dispatch path cares about. Returns `Err` for anything that
/// is not a structurally valid module.
pub fn reflect(spv: &[u8]) -> Result<Reflection, String> {
    if spv.len() < 20 {
        return Err(format!("too short to be SPIR-V: {} bytes", spv.len()));
    }
    if spv.len() % 4 != 0 {
        return Err(format!("not word-aligned: {} bytes", spv.len()));
    }
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words[0] != SPIRV_MAGIC {
        return Err(format!("bad magic 0x{:08x} (expected 0x07230203)", words[0]));
    }

    let mut r = Reflection {
        version: (((words[1] >> 16) & 0xff) as u8, ((words[1] >> 8) & 0xff) as u8),
        ..Default::default()
    };

    // id → decoration → operands
    let mut decorations: HashMap<u32, HashMap<u32, Vec<u32>>> = HashMap::new();
    // (struct id, member) → decoration → operands
    let mut member_decorations: HashMap<(u32, u32), HashMap<u32, Vec<u32>>> = HashMap::new();
    let mut types: HashMap<u32, Ty> = HashMap::new();
    let mut struct_members: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut constants: HashMap<u32, u32> = HashMap::new();
    // (result type, result id, storage class)
    let mut variables: Vec<(u32, u32, u32)> = Vec::new();
    let mut exec_modes: Vec<(u32, u32, Vec<u32>)> = Vec::new();

    let mut i = 5usize;
    while i < words.len() {
        let word0 = words[i];
        let count = (word0 >> 16) as usize;
        let op = (word0 & 0xffff) as u16;
        if count == 0 {
            return Err(format!("zero-length instruction at word {i}"));
        }
        if i + count > words.len() {
            return Err(format!(
                "instruction at word {i} claims {count} words but only {} remain",
                words.len() - i
            ));
        }
        let ops = &words[i + 1..i + count];

        match op {
            OP_ENTRY_POINT if ops.len() >= 3 => {
                r.entry_points.push((ops[0], decode_string(&ops[2..])));
            }
            OP_EXECUTION_MODE | OP_EXECUTION_MODE_ID if ops.len() >= 2 => {
                exec_modes.push((ops[0], ops[1], ops[2..].to_vec()));
            }
            OP_DECORATE if ops.len() >= 2 => {
                decorations
                    .entry(ops[0])
                    .or_default()
                    .insert(ops[1], ops[2..].to_vec());
            }
            OP_MEMBER_DECORATE if ops.len() >= 3 => {
                member_decorations
                    .entry((ops[0], ops[1]))
                    .or_default()
                    .insert(ops[2], ops[3..].to_vec());
            }
            OP_TYPE_INT if ops.len() >= 2 => {
                types.insert(ops[0], Ty::Scalar(ops[1].div_ceil(8)));
            }
            OP_TYPE_FLOAT if ops.len() >= 2 => {
                types.insert(ops[0], Ty::Scalar(ops[1].div_ceil(8)));
            }
            OP_TYPE_VECTOR if ops.len() >= 3 => {
                types.insert(ops[0], Ty::Vector(ops[1], ops[2]));
            }
            OP_TYPE_MATRIX if ops.len() >= 3 => {
                types.insert(ops[0], Ty::Matrix(ops[1], ops[2]));
            }
            OP_TYPE_ARRAY if ops.len() >= 3 => {
                types.insert(ops[0], Ty::Array(ops[1], ops[2]));
            }
            OP_TYPE_RUNTIME_ARRAY if ops.len() >= 2 => {
                types.insert(ops[0], Ty::RuntimeArray(ops[1]));
            }
            OP_TYPE_STRUCT if !ops.is_empty() => {
                types.insert(ops[0], Ty::Struct);
                struct_members.insert(ops[0], ops[1..].to_vec());
            }
            OP_TYPE_POINTER if ops.len() >= 3 => {
                types.insert(ops[0], Ty::Pointer(ops[1], ops[2]));
            }
            OP_CONSTANT | OP_SPEC_CONSTANT if ops.len() >= 3 => {
                // Only single-word (<=32-bit) constants matter here: array
                // lengths and workgroup dimensions are all 32-bit.
                constants.insert(ops[1], ops[2]);
            }
            OP_VARIABLE if ops.len() >= 3 => {
                variables.push((ops[0], ops[1], ops[2]));
            }
            _ => {}
        }
        i += count;
    }

    // Descriptor bindings: every id carrying a Binding decoration.
    for (id, decs) in &decorations {
        let Some(binding) = decs.get(&DEC_BINDING).and_then(|v| v.first()).copied() else {
            continue;
        };
        // An undecorated set defaults to 0 in GLSL.
        let set = decs
            .get(&DEC_DESCRIPTOR_SET)
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0);
        let _ = id;
        r.bindings.push((set, binding));
    }
    // A binding may be declared by SEVERAL variables — the ggml-derived
    // shaders routinely alias one storage buffer under two block types
    // (`layout(binding=2) buffer Activ {float x[];}` +
    //  `layout(binding=2) buffer ActivV4 {vec4 xv4[];}` in
    //  mul_mat_vec_fp8_fast.comp, or the A/A_PACKED16/A_PACKED32 trio in
    // generic_binary_head.glsl). That is one descriptor, so dedupe.
    r.bindings.sort_unstable();
    r.bindings.dedup();

    // Push-constant block: the (single) PushConstant OpVariable, its pointee
    // struct, sized from the member Offset decorations glslang always emits.
    for (result_type, _id, sc) in &variables {
        if *sc != SC_PUSH_CONSTANT {
            continue;
        }
        let Some(Ty::Pointer(_, pointee)) = types.get(result_type).copied() else {
            return Err("PushConstant variable has a non-pointer result type".to_string());
        };
        let size = size_of_ty(pointee, &types, &struct_members, &member_decorations, &constants, &decorations)
            .ok_or_else(|| "could not size the push-constant block type".to_string())?;
        r.push_constant_size = Some(size);
    }

    // Workgroup size.
    for (_target, mode, lits) in &exec_modes {
        if *mode == EXEC_MODE_LOCAL_SIZE && lits.len() >= 3 {
            r.local_size = Some((lits[0], lits[1], lits[2]));
        } else if *mode == EXEC_MODE_LOCAL_SIZE_ID {
            r.local_size_is_spec = true;
            if lits.len() >= 3 {
                if let (Some(x), Some(y), Some(z)) = (
                    constants.get(&lits[0]),
                    constants.get(&lits[1]),
                    constants.get(&lits[2]),
                ) {
                    r.local_size = Some((*x, *y, *z));
                }
            }
        }
    }
    // `layout(local_size_x_id = 0)` lowers to a spec-constant composite tagged
    // `BuiltIn WorkgroupSize` rather than a LocalSizeId execution mode.
    if decorations
        .values()
        .any(|d| d.get(&DEC_BUILT_IN).and_then(|v| v.first()) == Some(&BUILTIN_WORKGROUP_SIZE))
    {
        r.local_size_is_spec = true;
    }

    Ok(r)
}

/// Byte size of a type, using the explicit layout decorations glslang emits
/// for interface blocks. `None` when the type is not one we can size.
fn size_of_ty(
    id: u32,
    types: &HashMap<u32, Ty>,
    struct_members: &HashMap<u32, Vec<u32>>,
    member_decorations: &HashMap<(u32, u32), HashMap<u32, Vec<u32>>>,
    constants: &HashMap<u32, u32>,
    decorations: &HashMap<u32, HashMap<u32, Vec<u32>>>,
) -> Option<u32> {
    match types.get(&id)? {
        Ty::Scalar(w) => Some(*w),
        Ty::Vector(comp, n) => {
            let cw = size_of_ty(*comp, types, struct_members, member_decorations, constants, decorations)?;
            Some(cw * n)
        }
        Ty::Matrix(col, n) => {
            if let Some(stride) = decorations
                .get(&id)
                .and_then(|d| d.get(&DEC_MATRIX_STRIDE))
                .and_then(|v| v.first())
            {
                return Some(stride * n);
            }
            let cw = size_of_ty(*col, types, struct_members, member_decorations, constants, decorations)?;
            Some(cw * n)
        }
        Ty::Array(elem, len_id) => {
            let len = *constants.get(len_id)?;
            let stride = decorations
                .get(&id)
                .and_then(|d| d.get(&DEC_ARRAY_STRIDE))
                .and_then(|v| v.first())
                .copied();
            let elem_size = match stride {
                Some(s) => s,
                None => size_of_ty(*elem, types, struct_members, member_decorations, constants, decorations)?,
            };
            Some(elem_size * len)
        }
        // A runtime array has no static size; it can never appear in a
        // push-constant block (Vulkan forbids it), so treat it as unsizeable.
        Ty::RuntimeArray(_) => None,
        Ty::Struct => {
            let members = struct_members.get(&id)?;
            let mut end = 0u32;
            for (idx, mty) in members.iter().enumerate() {
                let off = member_decorations
                    .get(&(id, idx as u32))
                    .and_then(|d| d.get(&DEC_OFFSET))
                    .and_then(|v| v.first())
                    .copied()?;
                let sz = size_of_ty(*mty, types, struct_members, member_decorations, constants, decorations)?;
                end = end.max(off + sz);
            }
            Some(end)
        }
        Ty::Pointer(..) => None,
    }
}

/// Decode a SPIR-V literal string (NUL-terminated, packed little-endian).
fn decode_string(words: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    'outer: for w in words {
        for b in w.to_le_bytes() {
            if b == 0 {
                break 'outer;
            }
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

// ── The invariants ─────────────────────────────────────────────────────────

/// I1: the module is a compute shader with a `main` entry point, and its
/// SPIR-V version matches what the instance is created with.
///
/// Evidence: `pipeline.rs` names the entry point `c"main"` unconditionally
/// (`compile_one_with_spec`, `compile_with_spec_timeout`) and always builds a
/// `ComputePipelineCreateInfo` with `ShaderStageFlags::COMPUTE`. A module with
/// a differently-named entry point fails `vkCreateComputePipelines` at load.
/// `device.rs` requests `vk::API_VERSION_1_3` and `scripts/compile_shaders.sh`
/// compiles `--target-env vulkan1.3` (SPIR-V 1.6); a module built for a newer
/// target would be rejected by the driver.
pub fn check_entry_point(r: &Reflection) -> Result<(), String> {
    if r.entry_points.len() != 1 {
        return Err(format!(
            "expected exactly 1 OpEntryPoint, found {} ({:?}); pipeline.rs always \
             creates the pipeline with entry point \"main\"",
            r.entry_points.len(),
            r.entry_points
        ));
    }
    let (model, name) = &r.entry_points[0];
    if *model != EXEC_MODEL_GL_COMPUTE {
        return Err(format!(
            "entry point \"{name}\" has execution model {model}, expected GLCompute ({EXEC_MODEL_GL_COMPUTE}); \
             pipeline.rs only ever builds COMPUTE pipelines"
        ));
    }
    if name != "main" {
        return Err(format!(
            "entry point is named \"{name}\", but pipeline.rs hardcodes c\"main\" \
             (vkCreateComputePipelines would fail on device)"
        ));
    }
    if r.version != (1, 6) {
        return Err(format!(
            "SPIR-V version {}.{}, expected 1.6 (scripts/compile_shaders.sh uses \
             --target-env vulkan1.3 and device.rs requests API_VERSION_1_3)",
            r.version.0, r.version.1
        ));
    }
    Ok(())
}

/// I2a: every descriptor binding is in set 0, below `MAX_BINDINGS`.
///
/// Evidence: `PipelineCache::new` creates exactly ONE `DescriptorSetLayout`,
/// with bindings `(0..MAX_BINDINGS)`, and `compute.rs::record_to` binds it with
/// `cmd_bind_descriptor_sets(.., first_set = 0, &[ds], ..)`. A binding in set 1
/// — or at index >= MAX_BINDINGS — has no descriptor bound at all. `record_to`
/// additionally hard-errors above `MAX_BINDINGS` buffers.
pub fn check_bindings(r: &Reflection) -> Result<(), String> {
    let max = crate::pipeline::MAX_BINDINGS;
    for (set, binding) in &r.bindings {
        if *set != 0 {
            return Err(format!(
                "binding (set={set}, binding={binding}) is not in set 0; compute.rs \
                 binds a single descriptor set at first_set=0, so a set-{set} \
                 binding would have no descriptor bound"
            ));
        }
        if *binding >= max {
            return Err(format!(
                "binding {binding} >= MAX_BINDINGS ({max}); pipeline.rs's descriptor \
                 set layout only declares bindings 0..{}",
                max - 1
            ));
        }
    }
    Ok(())
}

/// I2b: the bindings are contiguous from 0 — the invariant the dispatch path
/// relies on *positionally*.
///
/// Evidence: `compute.rs::record_to` writes the caller's buffer slice by INDEX:
///
/// ```text
/// for (i, b) in buffers.iter().enumerate() { buffer_infos[i] = ...buffer(b.buffer)... }
/// for i in 0..n { writes[i] = ...dst_set(ds).dst_binding(i as u32)... }
/// ```
///
/// so `buffers[i]` always lands on binding `i`. A shader that declared, say,
/// bindings {0, 1, 3} would read binding 3 while the host wrote binding 2 —
/// undefined descriptor, garbage or hang, with nothing failing host-side.
/// `record_to_off` does the identical positional mapping.
pub fn check_bindings_contiguous(r: &Reflection) -> Result<(), String> {
    check_bindings(r)?;
    let got: Vec<u32> = r.bindings.iter().map(|(_, b)| *b).collect();
    let expected: Vec<u32> = (0..got.len() as u32).collect();
    if got != expected {
        return Err(format!(
            "descriptor bindings are not contiguous from 0: declared {got:?}, expected \
             {expected:?}. compute.rs::record_to writes buffers[i] to dst_binding(i), \
             so a hole silently misroutes every buffer after it"
        ));
    }
    Ok(())
}

/// Modules that deliberately leave a hole in their binding numbering, with the
/// reason. Anything NOT on this list must be contiguous (see
/// `check_bindings_contiguous`); a newly-holed shader therefore fails the gate
/// instead of reaching a GPU.
const BINDING_HOLE_ALLOWLIST: &[(&str, &str)] = &[
    // shaders/rms_norm.comp, `#if RMS_NORM_ROPE_FUSION` branch: "Binding 2 is
    // not used" — the rms_norm→rope handoff goes through shared memory, so the
    // fused variant declares 0,1,3,4,5,6. Upstream (llama.cpp) shape; this
    // crate compiles it but never dispatches it (no `rms_norm_mul_rope`
    // reference anywhere in src/), so no `record_to` call is exposed to the
    // hole. If it is ever dispatched, the caller must pass a filler buffer at
    // index 2.
    ("rms_norm_mul_rope_f32_f32", "rms_norm.comp: binding 2 intentionally unused"),
    ("rms_norm_mul_rope_f32_f16", "rms_norm.comp: binding 2 intentionally unused"),
];

/// I3: the declared push-constant block fits in the pipeline layout's range.
///
/// Evidence: every `PipelineLayoutCreateInfo` in `pipeline.rs` declares one
/// `PushConstantRange { offset: 0, size: 128 }`, and `compute.rs` pushes at
/// offset 0. A block larger than the range is a validation error at pipeline
/// creation, and the tail bytes would never be written.
pub fn check_push_constant_range(r: &Reflection) -> Result<(), String> {
    if let Some(sz) = r.push_constant_size {
        if sz > PUSH_CONSTANT_RANGE_BYTES {
            return Err(format!(
                "push-constant block is {sz} bytes, exceeding the \
                 PushConstantRange size pipeline.rs declares ({PUSH_CONSTANT_RANGE_BYTES})"
            ));
        }
    }
    Ok(())
}

/// I4: the workgroup size is either a literal `LocalSize` within the Vulkan
/// guaranteed minimum (1024 invocations), or specialization-constant driven —
/// in which case `pipeline.rs` supplies `BLOCK_SIZE` at pipeline creation.
pub fn check_local_size(r: &Reflection) -> Result<(), String> {
    match (r.local_size, r.local_size_is_spec) {
        (Some((x, y, z)), false) => {
            let total = x as u64 * y as u64 * z as u64;
            if total == 0 {
                return Err("LocalSize has a zero dimension".to_string());
            }
            if total > 1024 {
                return Err(format!(
                    "LocalSize {x}x{y}x{z} = {total} invocations exceeds the Vulkan \
                     guaranteed maxComputeWorkGroupInvocations (1024)"
                ));
            }
            Ok(())
        }
        (_, true) => Ok(()),
        (None, false) => Err(
            "no LocalSize execution mode and no spec-constant WorkgroupSize — the \
             workgroup size is undefined"
                .to_string(),
        ),
    }
}

/// Modules whose push-constant block does NOT fit the 128-byte
/// `PushConstantRange` that `pipeline.rs` declares — i.e. modules that cannot
/// be fed correctly on a device with the Vulkan guaranteed-minimum
/// `maxPushConstantsSize` (which is what GFX1013/RADV reports).
///
/// All four are upstream-ggml shaders (`multi_add.comp`'s `uint nb[12][4]`
/// stride table = 4 + 192 + 4 = 212 bytes; `rms_norm.comp`'s
/// `RMS_NORM_ROPE_FUSION` branch pulls in `rope_params.glsl` for 228 bytes).
/// Upstream llama.cpp sizes its push-constant range from the device limit;
/// this fork hardcodes 128. They are compiled and registered but NEVER
/// dispatched — no `add_rms`/`rms_norm_mul_rope` string appears anywhere in
/// `src/` — so nothing reads the truncated tail today. They would trip
/// `VUID-VkComputePipelineCreateInfo-layout` under the validation layers.
///
/// The list is exact (see `oversize_push_constant_set_is_exactly_the_known_four`):
/// a NEW shader with an oversize block fails the gate.
const PUSH_CONSTANT_OVERSIZE_ALLOWLIST: &[&str] = &[
    "add_rms_f32_f32_f32",
    "add_rms_f32_f32_f16",
    "rms_norm_mul_rope_f32_f32",
    "rms_norm_mul_rope_f32_f16",
];

/// Run every module-level invariant for the shader named `name`.
pub fn check_module(name: &str, r: &Reflection) -> Result<(), String> {
    check_entry_point(r)?;
    if BINDING_HOLE_ALLOWLIST.iter().any(|(n, _)| *n == name) {
        check_bindings(r)?;
    } else {
        check_bindings_contiguous(r)?;
    }
    if !PUSH_CONSTANT_OVERSIZE_ALLOWLIST.contains(&name) {
        check_push_constant_range(r)?;
    }
    check_local_size(r)?;
    Ok(())
}

// ── kernel → (Rust push-constant producer, binding count) ──────────────────

/// The Rust helper in `push_constants.rs` that feeds each required kernel, as
/// a name plus the byte length it produces, together with the number of
/// storage buffers the shader declares.
///
/// `push_constants.rs` has no `#[repr(C)]` structs — every push-constant block
/// is produced by a helper that serialises into a `Vec<u8>` — so the Rust-side
/// size is that helper's output length, evaluated here by calling the helper.
struct KernelContract {
    kernel: &'static str,
    /// Name of the `push_constants.rs` helper (for the failure message).
    pc_helper: &'static str,
    /// Byte length that helper produces.
    pc_rust_bytes: u32,
    /// Byte length the shader's push-constant block is expected to reflect as.
    /// Normally identical to `pc_rust_bytes`; a smaller value means the shader
    /// reads a strict PREFIX of what the helper pushes (legal — the tail is
    /// ignored — but always a divergence worth naming), and requires
    /// `divergence` to explain it. Larger is never allowed: the shader would
    /// read bytes the host never wrote.
    pc_shader_bytes: u32,
    /// Why `pc_shader_bytes != pc_rust_bytes`, when they differ.
    divergence: Option<&'static str>,
    /// Storage buffers the shader binds (= the length of the `buffers` slice
    /// the dispatch site must pass to `record_to`).
    bindings: u32,
}

fn contracts() -> Vec<KernelContract> {
    use crate::push_constants as pc;

    // Evaluate each helper rather than hardcoding a number, so a change to a
    // helper's layout shows up here instead of silently drifting.
    let mlx4 = pc::matvec_mlx4_pc(64, 64, 64).len() as u32; // {k,n,gs,packed_off,sb_off}
    let nvfp4_e4m3 = pc::matvec_nvfp4_e4m3_pc(64, 64, 16, 1.0).len() as u32; // + float global
    let fp8 = pc::matvec_fp8_pc(64, 64, false).len() as u32;
    let cols2 = pc::matvec_cols_pc2(64, 64).len() as u32;
    let sdpa = pc::sdpa_pc(1, 1, 1, 64, 16, 1, 1.0, 0, 0).len() as u32;
    let unary = pc::ew_unary_pc(1).len() as u32;

    let c = |kernel, pc_helper, pc_rust_bytes, bindings| KernelContract {
        kernel,
        pc_helper,
        pc_rust_bytes,
        pc_shader_bytes: pc_rust_bytes,
        divergence: None,
        bindings,
    };

    vec![
        // mlx4 (4-bit affine) family — all share matvec_mlx4_pc's 5-uint block
        // ({ncols, nrows, group_size, packed_off, sb_off}); the `_batched`
        // siblings say so verbatim in their .comp ("kept so matvec_mlx4_pc's
        // 5-uint layout is reused verbatim") and add a `meta[]` binding.
        c("mul_mat_vec_mlx4_f32_f32",               "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx4_cols",                  "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx4w8_f32_f32",             "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx4w16_f32_f32",            "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx4w8sg_f32_f32",           "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx4repack_f32_f32",         "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx4repack_batched_f32_f32", "matvec_mlx4_pc",       mlx4,       6),
        // mlx2 / mlx6 / mlx8 — same 5-uint block (see each .comp's `parameter`).
        c("mul_mat_vec_mlx2repack_f32_f32",         "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx2repack_batched_f32_f32", "matvec_mlx4_pc",       mlx4,       6),
        c("mul_mat_vec_mlx6_f32_f32",               "matvec_mlx4_pc",       mlx4,       5),
        c("mul_mat_vec_mlx8_f32_f32",               "matvec_mlx4_pc",       mlx4,       5),
        // nvfp4: the f32-fold kernels reuse the 5-uint block; the E4M3-resident
        // ones append `float global` (push_constants::nvfp4_dispatch routes
        // between them, and documents that both share the same 4 bindings).
        c("mul_mat_vec_nvfp4_f32_f32",              "matvec_mlx4_pc",       mlx4,       4),
        c("mul_mat_vec_nvfp4repack_f32_f32",        "matvec_mlx4_pc",       mlx4,       4),
        c("mul_mat_vec_nvfp4_e4m3_f32_f32",         "matvec_nvfp4_e4m3_pc", nvfp4_e4m3, 4),
        c("mul_mat_vec_nvfp4_e4m3repack_f32_f32",   "matvec_nvfp4_e4m3_pc", nvfp4_e4m3, 4),
        // fp8 ({k, n, scale_per_row, packed_off, sb_off}).
        c("mul_mat_vec_fp8_f32_f32",                "matvec_fp8_pc",        fp8,        4),
        c("mul_mat_vec_fp8fast_f32_f32",            "matvec_fp8_pc",        fp8,        4),
        c("mul_mat_vec_fp8repack_f32_f32",          "matvec_fp8_pc",        fp8,        4),
        // Column-batched dequant matvec (standalone kernels, 2-uint block —
        // matvec_cols_pc2's doc comment names exactly these two shaders).
        c("mul_mat_vec_q8_0_cols",                  "matvec_cols_pc2",      cols2,      3),
        c("mul_mat_vec_f16_cols",                   "matvec_cols_pc2",      cols2,      3),
        // Subgroup paged decode attention. f32 takes sdpa_pc's full 11 words;
        // f16 stops one word short — see `divergence`.
        c("paged_attn_decode_f32_sg",               "sdpa_pc",              sdpa,       4),
        KernelContract {
            kernel: "paged_attn_decode_f16_sg",
            pc_helper: "sdpa_pc",
            pc_rust_bytes: sdpa,
            pc_shader_bytes: sdpa - 4,
            divergence: Some(
                "paged_attn_decode_f16_sg.comp's push block ends at `window_start` \
                 and has NO `ring_capacity` word, unlike its f32 twin. sdpa_pc \
                 pushes all 11 words; the f16 kernel ignores the 11th, i.e. the \
                 f16-KV decode path silently does ABSOLUTE block-table addressing \
                 even when the caller asks for ring addressing (ring_capacity > 0). \
                 Harmless while the ring is only used with f32 KV; a real \
                 mis-addressing bug the moment it is not.",
            ),
            bindings: 4,
        },
        // Elementwise (generic_head.glsl: {KX, KY, param1..4}).
        c("relu2_f32",                              "ew_unary_pc",          unary,      2),
    ]
}

/// I3b/I4b: the reflected push-constant block size equals what the Rust helper
/// that feeds this kernel serialises, and the reflected binding count equals
/// the number of buffers its dispatch site passes to `record_to`. Returns one
/// message per violation (empty = the contract holds).
fn check_contract(c: &KernelContract, r: &Reflection) -> Vec<String> {
    let mut out = Vec::new();

    // A shader must never declare MORE than the host pushes: those trailing
    // bytes would be read but never written.
    if c.pc_shader_bytes > c.pc_rust_bytes {
        out.push(format!(
            "{}: contract claims the shader block ({}) is larger than what \
             push_constants::{} pushes ({}) — that is never legal",
            c.kernel, c.pc_shader_bytes, c.pc_helper, c.pc_rust_bytes
        ));
    }
    if c.pc_shader_bytes != c.pc_rust_bytes && c.divergence.is_none() {
        out.push(format!(
            "{}: shader block ({}) differs from push_constants::{} ({}) with no \
             documented reason",
            c.kernel, c.pc_shader_bytes, c.pc_helper, c.pc_rust_bytes
        ));
    }
    match r.push_constant_size {
        Some(sz) if sz == c.pc_shader_bytes => {}
        Some(sz) => out.push(format!(
            "{}: shader declares a {sz}-byte push-constant block but the dispatch \
             contract expects {} (push_constants::{} produces {} bytes) — \
             vkCmdPushConstants would leave {} byte(s) of the block undefined (or \
             silently ignore that many)",
            c.kernel,
            c.pc_shader_bytes,
            c.pc_helper,
            c.pc_rust_bytes,
            (sz as i64 - c.pc_shader_bytes as i64).abs()
        )),
        None => out.push(format!(
            "{}: shader declares NO push-constant block but push_constants::{} \
             produces {} bytes for it",
            c.kernel, c.pc_helper, c.pc_rust_bytes
        )),
    }
    let n = r.bindings.len() as u32;
    if n != c.bindings {
        out.push(format!(
            "{}: shader declares {n} descriptor binding(s) but the dispatch contract \
             binds {} buffer(s) — compute.rs::record_to maps buffers[i] to binding i, \
             so the counts must agree",
            c.kernel, c.bindings
        ));
    }
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry_tests::REQUIRED_QUANT_KERNELS;

    fn shaders() -> HashMap<String, Vec<u8>> {
        crate::include_all_shaders()
    }

    /// Every module in the shipped registry — not just the required quant
    /// kernels — must parse and satisfy the module-level invariants.
    #[test]
    fn all_registered_modules_satisfy_dispatch_invariants() {
        let mut failures: Vec<String> = Vec::new();
        let mut checked = 0usize;
        for (name, spv) in shaders() {
            match reflect(&spv) {
                Ok(r) => {
                    checked += 1;
                    if let Err(e) = check_module(&name, &r) {
                        failures.push(format!("{name}: {e}"));
                    }
                }
                Err(e) => failures.push(format!("{name}: unparseable SPIR-V: {e}")),
            }
        }
        assert!(checked > 0, "no shaders in the registry to reflect");
        assert!(
            failures.is_empty(),
            "{} of {checked} module(s) violate a dispatch invariant:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }

    /// The two allowlists above must stay EXACT: they are the documented
    /// exceptions to invariants the dispatch path otherwise relies on, so a
    /// newly-added shader that trips either one has to be looked at (and a
    /// fixed/removed one has to be de-listed) rather than silently inheriting
    /// the exemption.
    #[test]
    fn oversize_push_constant_set_is_exactly_the_known_four() {
        let mut oversize: Vec<String> = Vec::new();
        let mut holed: Vec<String> = Vec::new();
        for (name, spv) in shaders() {
            let r = reflect(&spv).expect("parses");
            if check_push_constant_range(&r).is_err() {
                oversize.push(name.clone());
            }
            if check_bindings_contiguous(&r).is_err() {
                holed.push(name);
            }
        }
        oversize.sort();
        holed.sort();
        let mut expect_oversize: Vec<String> = PUSH_CONSTANT_OVERSIZE_ALLOWLIST
            .iter()
            .map(|s| s.to_string())
            .collect();
        expect_oversize.sort();
        assert_eq!(
            oversize, expect_oversize,
            "the set of shaders whose push-constant block exceeds pipeline.rs's \
             128-byte PushConstantRange changed; see PUSH_CONSTANT_OVERSIZE_ALLOWLIST"
        );
        let mut expect_holed: Vec<String> = BINDING_HOLE_ALLOWLIST
            .iter()
            .map(|(n, _)| n.to_string())
            .collect();
        expect_holed.sort();
        assert_eq!(
            holed, expect_holed,
            "the set of shaders with a non-contiguous binding numbering changed; \
             see BINDING_HOLE_ALLOWLIST"
        );
    }

    /// Every kernel named in `REQUIRED_QUANT_KERNELS` has a contract entry, and
    /// every contract entry names a required kernel. Keeps the two lists from
    /// drifting apart silently (a new required kernel must state its
    /// push-constant producer).
    #[test]
    fn contract_table_covers_every_required_kernel() {
        let contracts = contracts();
        let covered: std::collections::HashSet<&str> =
            contracts.iter().map(|c| c.kernel).collect();
        let required: std::collections::HashSet<&str> =
            REQUIRED_QUANT_KERNELS.iter().copied().collect();
        let missing: Vec<&&str> = required.iter().filter(|k| !covered.contains(**k)).collect();
        let extra: Vec<&&str> = covered.iter().filter(|k| !required.contains(**k)).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "contract table out of sync with REQUIRED_QUANT_KERNELS: \
             missing {missing:?}, unexpected {extra:?}"
        );
        assert_eq!(contracts.len(), REQUIRED_QUANT_KERNELS.len());
    }

    /// The heart of the gate: each required kernel's *reflected* push-constant
    /// block size must equal the byte length the Rust helper that feeds it
    /// produces, and its binding count must equal the number of buffers the
    /// dispatch site passes.
    #[test]
    fn required_kernels_match_their_rust_push_constants() {
        let map = shaders();
        let mut failures: Vec<String> = Vec::new();
        for c in contracts() {
            let Some(spv) = map.get(c.kernel) else {
                // registry_tests::quant_matvec_kernels_are_registered owns the
                // "is it present" gate; don't duplicate its failure here.
                continue;
            };
            let r = match reflect(spv) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("{}: unparseable: {e}", c.kernel));
                    continue;
                }
            };
            failures.extend(check_contract(&c, &r));
        }
        assert!(
            failures.is_empty(),
            "{} push-constant/binding contract violation(s):\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }

    // ── Fail-injection: prove the gates can actually fail ──────────────────

    /// Rewrite the operand of the last `OpDecorate <id> Binding <n>` in a
    /// module, simulating a `.comp` whose bindings grew a hole.
    fn corrupt_last_binding(spv: &[u8], new_value: u32) -> Vec<u8> {
        let mut words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut target: Option<usize> = None;
        let mut i = 5;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            let op = (words[i] & 0xffff) as u16;
            if count == 0 {
                break;
            }
            if op == OP_DECORATE && count >= 4 && words[i + 2] == DEC_BINDING {
                target = Some(i + 3);
            }
            i += count;
        }
        let idx = target.expect("module has at least one Binding decoration");
        words[idx] = new_value;
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A real module, byte-corrupted to have a binding hole, must be REJECTED
    /// by `check_bindings` with a message that names the hole.
    #[test]
    fn fail_injection_binding_hole_is_caught() {
        let map = shaders();
        let spv = map
            .get("mul_mat_vec_mlx4_f32_f32")
            .expect("mlx4 matvec present");
        // Sanity: the pristine module passes.
        let good = reflect(spv).expect("pristine module parses");
        check_module("mul_mat_vec_mlx4_f32_f32", &good)
            .expect("pristine module satisfies every invariant");
        assert_eq!(good.bindings.len(), 5);

        // Bindings 0..4 -> 0,1,2,3,9: a hole at 4.
        let bad_bytes = corrupt_last_binding(spv, 9);
        let bad = reflect(&bad_bytes).expect("corrupted module still parses");
        let err = check_bindings_contiguous(&bad).expect_err("binding hole must be rejected");
        assert!(
            err.contains("not contiguous") && err.contains('9'),
            "unhelpful message for a binding hole: {err}"
        );

        // And a binding past the descriptor-set layout's last slot.
        let oob = corrupt_last_binding(spv, crate::pipeline::MAX_BINDINGS + 3);
        let oob_r = reflect(&oob).expect("corrupted module still parses");
        let err = check_bindings(&oob_r).expect_err("out-of-range binding must be rejected");
        assert!(err.contains("MAX_BINDINGS"), "unhelpful message: {err}");
    }

    /// A push-constant size mismatch must be reported by the SAME
    /// `check_contract` the production test runs, with both sizes named.
    #[test]
    fn fail_injection_push_constant_size_mismatch_is_caught() {
        let map = shaders();
        let spv = map.get("mul_mat_vec_fp8_f32_f32").expect("fp8 matvec present");
        let r = reflect(spv).expect("parses");
        let real = r.push_constant_size.expect("fp8 matvec has a push block");
        assert_eq!(real, crate::push_constants::matvec_fp8_pc(64, 64, false).len() as u32);

        // (a) The Rust helper gains a word the shader does not have — exactly
        // what adding a field to a `matvec_*_pc` helper and forgetting the
        // `.comp` looks like.
        let wrong = KernelContract {
            kernel: "mul_mat_vec_fp8_f32_f32",
            pc_helper: "matvec_fp8_pc",
            pc_rust_bytes: real + 4,
            pc_shader_bytes: real + 4,
            divergence: None,
            bindings: 4,
        };
        let errs = check_contract(&wrong, &r);
        assert_eq!(errs.len(), 1, "expected exactly one violation, got {errs:?}");
        assert!(
            errs[0].contains(&format!("{real}-byte")) && errs[0].contains("matvec_fp8_pc"),
            "unhelpful message: {}",
            errs[0]
        );

        // (b) A shorter shader block with NO documented divergence must be
        // rejected even though under-reading is technically harmless.
        let undocumented = KernelContract {
            kernel: "paged_attn_decode_f16_sg",
            pc_helper: "sdpa_pc",
            pc_rust_bytes: 44,
            pc_shader_bytes: 40,
            divergence: None,
            bindings: 4,
        };
        let f16 = reflect(map.get("paged_attn_decode_f16_sg").expect("present")).expect("parses");
        let errs = check_contract(&undocumented, &f16);
        assert!(
            errs.iter().any(|e| e.contains("no documented reason")),
            "an undocumented push-constant divergence must be flagged: {errs:?}"
        );

        // (c) The binding-count half of the contract.
        let wrong_bindings = KernelContract { bindings: 3, ..wrong };
        let errs = check_contract(&wrong_bindings, &r);
        assert!(
            errs.iter().any(|e| e.contains("descriptor binding(s)")),
            "a binding-count mismatch must be flagged: {errs:?}"
        );
    }

    /// The header/entry-point gate must reject a truncated or mis-tagged module.
    #[test]
    fn fail_injection_bad_header_is_caught() {
        let map = shaders();
        let spv = map.get("relu2_f32").expect("relu2 present").clone();

        let mut bad_magic = spv.clone();
        bad_magic[0] ^= 0xff;
        let err = reflect(&bad_magic).expect_err("bad magic must be rejected");
        assert!(err.contains("bad magic"), "unhelpful message: {err}");

        // Claim a 1.5 module (what --target-env vulkan1.2 would emit): the
        // instance asks for Vulkan 1.3 / SPIR-V 1.6.
        let mut old_version = spv.clone();
        old_version[5] = 5; // version word = 0x0001_0500
        let r = reflect(&old_version).expect("still parses");
        let err = check_entry_point(&r).expect_err("wrong SPIR-V version must be rejected");
        assert!(err.contains("1.6"), "unhelpful message: {err}");

        // A module cut mid-instruction must be rejected, not silently
        // half-reflected (which would let a truncated .spv pass every gate
        // above by simply declaring nothing).
        let truncated = &spv[..spv.len() / 2];
        let err = reflect(truncated).expect_err("a truncated module must not reflect cleanly");
        assert!(err.contains("words"), "unhelpful message: {err}");

        // Not word-aligned at all.
        assert!(reflect(&spv[..spv.len() - 2]).is_err());
    }

    /// `check_bindings` rejects a set-1 binding: nothing binds descriptor set 1.
    #[test]
    fn fail_injection_wrong_descriptor_set_is_caught() {
        let r = Reflection {
            version: (1, 6),
            entry_points: vec![(EXEC_MODEL_GL_COMPUTE, "main".into())],
            bindings: vec![(0, 0), (1, 1)],
            ..Default::default()
        };
        let err = check_bindings(&r).expect_err("set-1 binding must be rejected");
        assert!(err.contains("set 0"), "unhelpful message: {err}");
    }

    /// Parser unit checks against a known module, so a silent parser
    /// regression (e.g. always returning zero bindings, which would make every
    /// gate above vacuously pass) fails loudly.
    #[test]
    fn parser_extracts_known_values() {
        let map = shaders();
        let spv = map.get("mul_mat_vec_q8_0_cols").expect("present");
        let r = reflect(spv).expect("parses");
        assert_eq!(r.version, (1, 6));
        assert_eq!(r.entry_points, vec![(EXEC_MODEL_GL_COMPUTE, "main".to_string())]);
        // shaders/mul_mat_vec_q8_0_cols.comp: bindings 0 (W), 1 (Activ), 2 (Dst)
        assert_eq!(r.bindings, vec![(0, 0), (0, 1), (0, 2)]);
        // ... and `{ uint ncols; uint nrows; }`
        assert_eq!(r.push_constant_size, Some(8));

        // relu2.comp: `layout(local_size_x = 512, ...)` — a literal LocalSize.
        let relu = reflect(map.get("relu2_f32").expect("present")).expect("parses");
        assert_eq!(relu.local_size, Some((512, 1, 1)));
        assert!(!relu.local_size_is_spec);
        // generic_head.glsl: { uint KX; uint KY; float param1..4 } = 24 bytes.
        assert_eq!(relu.push_constant_size, Some(24));

        // mul_mat_vec_mlx4repack uses layout(local_size_x_id = 0) — pipeline.rs
        // supplies BLOCK_SIZE as a specialization constant.
        let repack = reflect(map.get("mul_mat_vec_mlx4repack_f32_f32").expect("present"))
            .expect("parses");
        assert!(
            repack.local_size_is_spec,
            "mlx4repack's workgroup size is spec-constant driven (local_size_x_id = 0)"
        );
    }
}
