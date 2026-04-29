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
    println!("cargo:rerun-if-changed=shaders/");

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
    let spirv_src = Path::new(&manifest_dir).join("shaders").join("spirv");
    let out_spirv = Path::new(&out_dir).join("spirv");

    fs::create_dir_all(&out_spirv).expect("failed to create OUT_DIR/spirv");

    // Count already-compiled .spv files in OUT_DIR (incremental rebuild).
    let compiled: Vec<_> = fs::read_dir(&out_spirv)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("spv")
                })
                .collect()
        })
        .unwrap_or_default();

    if !compiled.is_empty() {
        println!(
            "cargo:warning=vllm-vulkan: {} SPIR-V shaders already in OUT_DIR, skipping compilation",
            compiled.len()
        );
        return;
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

    // Also copy to shaders/spirv/ so they're available for subsequent builds
    // without re-running the shader compiler.
    fs::create_dir_all(&spirv_src).ok();
    for entry in fs::read_dir(&out_spirv).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("spv") {
            let dst = spirv_src.join(path.file_name().unwrap());
            if !dst.exists() {
                fs::copy(&path, &dst).ok();
            }
        }
    }
}

// ─── macOS ────────────────────────────────────────────────────────────────────

fn link_macos() {
    let search_paths = ["/opt/homebrew/lib", "/usr/local/lib"];

    let mut linked = false;
    for lib_dir in &search_paths {
        if Path::new(lib_dir).join("libvulkan.dylib").exists() {
            println!("cargo:rustc-link-search=native={lib_dir}");
            println!("cargo:rustc-link-lib=dylib=vulkan");
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
