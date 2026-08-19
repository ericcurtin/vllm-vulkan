// SPDX-License-Identifier: Apache-2.0
//! Build script for vllm-vulkan.
//!
//! Platform routing:
//!   macOS (aarch64 / x86_64) — link against libvulkan.dylib installed by
//!     KosmicKrisp (Mesa/Zink software Vulkan driver for macOS).
//!
//!   Linux (x86_64 / aarch64) — link against the system libvulkan.so loader
//!     installed via libvulkan-dev (Debian/Ubuntu).

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=scripts/compile_shaders.sh");
    // Watch only shader SOURCES, not the spirv/ output subdir (watching the
    // whole shaders/ dir would self-trigger rebuilds from the compiled output).
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    if let Ok(entries) = fs::read_dir(Path::new(&manifest_dir).join("shaders")) {
        for entry in entries.flatten() {
            let p = entry.path();
            match p.extension().and_then(|e| e.to_str()) {
                Some("comp") | Some("glsl") => {
                    println!("cargo:rerun-if-changed={}", p.display());
                }
                _ => {}
            }
        }
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match target_os.as_str() {
        "macos" => link_macos(),
        "linux" => link_linux(),
        other => {
            println!(
                "cargo:warning=vllm-vulkan: unsupported target OS '{other}'. \
                 Only macOS and Linux are supported."
            );
        }
    }

    compile_shaders();
}

// ─── Shader compilation ───────────────────────────────────────────────────────

fn compile_shaders() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_spirv = Path::new(&out_dir).join("spirv");

    fs::create_dir_all(&out_spirv).expect("failed to create OUT_DIR/spirv");

    // The build script only reruns when shader sources change, so when it does
    // run, rebuild OUT_DIR's SPIR-V set from source.  Otherwise adding a new
    // shader can leave stale OUT_DIR contents and break include_bytes! below.
    for entry in fs::read_dir(&out_spirv).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("spv") {
            fs::remove_file(path).ok();
        }
    }

    // Run compile_shaders.sh to build the .spv files into OUT_DIR/spirv.
    let compile_script = Path::new(&manifest_dir)
        .join("scripts")
        .join("compile_shaders.sh");

    println!("cargo:warning=vllm-vulkan: compiling SPIR-V shaders...");

    let status = Command::new("bash")
        .arg(&compile_script)
        .arg(&out_spirv)
        .status();

    match status {
        Ok(s) if s.success() => {
            let count = fs::read_dir(&out_spirv)
                .map(|rd| rd.flatten().count())
                .unwrap_or(0);
            println!("cargo:warning=vllm-vulkan: compiled {count} SPIR-V shaders");
        }
        Ok(s) => {
            panic!(
                "compile_shaders.sh failed with exit code {s}.\n\
                 Install glslangValidator:\n\
                   Ubuntu/Debian: sudo apt-get install -y glslang-tools\n\
                   macOS:         brew install glslang"
            );
        }
        Err(e) => {
            panic!("failed to run compile_shaders.sh: {e}");
        }
    }

    stamp_glslang_version(&out_spirv);
    generate_registry(&out_spirv);
}

// ─── Shader registry generation (H5) ──────────────────────────────────────────
//
// Single source of truth for (a) which shaders are registered, (b) their
// bytes, and (c) their pipeline class. Consumed by `src/lib.rs`
// (`include_all_shaders`) and `src/pipeline.rs` (`PipelineCache::new`) via
// `include!(concat!(env!("OUT_DIR"), "/shader_registry.rs"))`.

/// Compiled-but-intentionally-unregistered shaders (present in
/// `scripts/compile_shaders.sh` output, but never referenced from `src/`).
const SKIP: &[&str] = &[
    "div_f32_f32_f32",
    "sub_f32_f32_f32",
    // Design A quant tiled-GEMM (PREFILL/M6) — compiled + verified but not yet
    // registered: they need BK=32 (spec constant_id 3) + a quant-aware prefill
    // dispatch, not the f16 MulMm class's BK=16 compile. See
    // scripts/compile_shaders.sh (Design A block) + quant-batched-matmul-impl.md.
    "matmul_q8_0_f32_fp32",
    "matmul_q4_k_f32_fp32",
    "matmul_q6_k_f32_fp32",
];

/// Registered alias name -> source `.spv` stem whose bytes it reuses.
/// (Gemma4 head_dim variants share the same flash_attn_f32_f16_f32 SPIR-V;
/// registered under distinct names so PipelineCache::new's flash_attn_ prefix
/// routing picks HSK/HSV=256/512 from the "_hs256"/"_hs512" suffix.)
const ALIASES: &[(&str, &str)] = &[
    ("flash_attn_f32_f16_f32_hs256", "flash_attn_f32_f16_f32"),
    ("flash_attn_f32_f16_f32_hs512", "flash_attn_f32_f16_f32"),
];

/// Mirrors `PipelineCache::new`'s name-based routing in src/pipeline.rs
/// EXACTLY (same order, same predicates) so the generated class always
/// matches what the runtime dispatch would have picked.
fn class_of(name: &str) -> &'static str {
    if name == "rms_norm_f32" {
        "RmsNorm"
    } else if name.ends_with("_subgroup") {
        "MatvecSubgroup"
    } else if name.starts_with("mul_mat_vec_") {
        "Matvec"
    } else if name.starts_with("matmul_") {
        "MulMm"
    } else if name.starts_with("flash_attn_") {
        "Flash"
    } else {
        "Plain"
    }
}

fn generate_registry(out_spirv: &Path) {
    let out_dir = env::var("OUT_DIR").unwrap();

    let mut stems: Vec<String> = fs::read_dir(out_spirv)
        .expect("read_dir OUT_DIR/spirv")
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("spv") {
                path.file_stem().map(|s| s.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .filter(|stem| !SKIP.contains(&stem.as_str()))
        .collect();
    stems.sort(); // deterministic generated output

    let mut out = String::new();
    out.push_str(
        "pub static SHADER_REGISTRY: &[(&str, crate::pipeline::ShaderClass, &[u8])] = &[\n",
    );
    for stem in &stems {
        let class = class_of(stem);
        let spv_path = out_spirv.join(format!("{stem}.spv"));
        out.push_str(&format!(
            "    (\"{stem}\", crate::pipeline::ShaderClass::{class}, include_bytes!(r\"{}\")),\n",
            spv_path.display()
        ));
    }
    for (alias, source) in ALIASES {
        assert!(
            stems.iter().any(|s| s == source),
            "H5 registry: alias '{alias}' source stem '{source}' not among compiled shaders"
        );
        // Aliases route through the same class rule as their own name (the
        // "_hsNNN" suffix still matches the flash_attn_ prefix).
        let class = class_of(alias);
        let spv_path = out_spirv.join(format!("{source}.spv"));
        out.push_str(&format!(
            "    (\"{alias}\", crate::pipeline::ShaderClass::{class}, include_bytes!(r\"{}\")),\n",
            spv_path.display()
        ));
    }
    out.push_str("];\n");

    fs::write(Path::new(&out_dir).join("shader_registry.rs"), out)
        .expect("write OUT_DIR/shader_registry.rs");
}

// Version stamp + warning is the pragmatic first step; a fully pinned shaderc
// build-dep is future hardening (H5-full territory).
const EXPECTED_GLSLANG: &str = "11:16.3.0"; // pin: update deliberately

fn stamp_glslang_version(out_spirv: &Path) {
    let ver = Command::new("glslangValidator")
        .arg("--version")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_default();
    if !ver.contains(EXPECTED_GLSLANG) {
        println!(
            "cargo:warning=vllm-vulkan: glslangValidator '{ver}' != pinned '{EXPECTED_GLSLANG}'; \
             embedded SPIR-V may differ across hosts"
        );
    }
    fs::write(out_spirv.join("GLSLANG_VERSION.txt"), ver).ok();
}

// ─── macOS ────────────────────────────────────────────────────────────────────

fn link_macos() {
    let home = env::var("HOME").unwrap_or_default();
    let home_local_lib = format!("{home}/.local/lib");
    let search_paths: Vec<String> = vec![
        home_local_lib,
        "/opt/homebrew/lib".to_string(),
        "/usr/local/lib".to_string(),
    ];

    let mut linked = false;
    for lib_dir in &search_paths {
        if Path::new(lib_dir.as_str()).join("libvulkan.dylib").exists() {
            println!("cargo:rustc-link-search=native={lib_dir}");
            println!("cargo:rustc-link-lib=dylib=vulkan");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
            linked = true;
            break;
        }
    }

    if !linked {
        println!(
            "cargo:warning=libvulkan.dylib not found. \
             Install KosmicKrisp: curl -fsSL https://raw.githubusercontent.com/ericcurtin/vllm-vulkan/main/install.sh | bash"
        );
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}

// ─── Linux ────────────────────────────────────────────────────────────────────

fn link_linux() {
    let search_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
    ];

    let found = search_paths
        .iter()
        .any(|dir| Path::new(dir).join("libvulkan.so").exists());

    if !found {
        println!(
            "cargo:warning=libvulkan.so not found. \
             Install libvulkan-dev: sudo apt-get install -y libvulkan-dev"
        );
    }

    for dir in &search_paths {
        if Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    println!("cargo:rustc-link-lib=dylib=vulkan");
}
