// SPDX-License-Identifier: Apache-2.0
//! Build script for vllm-vulkan.
//!
//! Platform routing:
//!   macOS (aarch64 / x86_64) — link against libvulkan.dylib installed by
//!     KosmicKrisp (Mesa/Zink software Vulkan driver for macOS).
//!     KosmicKrisp provides a standard libvulkan.dylib loader.
//!
//!   Linux (x86_64 / aarch64) — link against the system libvulkan.so loader
//!     installed via libvulkan-dev (Debian/Ubuntu) or vulkan-loader-devel
//!     (Fedora/RHEL).

use std::env;
use std::fs;
use std::path::Path;

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
                 Only macOS (via KosmicKrisp) and Linux are supported."
            );
        }
    }

    compile_shaders();
}

// ─── Shader compilation ───────────────────────────────────────────────────────

fn compile_shaders() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new(&manifest_dir).join("shaders");
    let spirv_dir = shader_dir.join("spirv");

    // Copy pre-compiled SPIR-V files from shaders/spirv/ to OUT_DIR so that
    // include_bytes! can find them at compile time.
    let out_spirv = Path::new(&out_dir).join("spirv");
    fs::create_dir_all(&out_spirv).expect("failed to create spirv output dir");

    let spv_count = fs::read_dir(&spirv_dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.path().extension().and_then(|x| x.to_str()) == Some("spv")
                })
                .count()
        })
        .unwrap_or(0);

    if spv_count == 0 {
        panic!(
            "vllm-vulkan: no pre-compiled SPIR-V shaders found in shaders/spirv/.\n\
             Run scripts/compile_shaders.sh to build them, or check out the\n\
             repository with the shader .spv files present."
        );
    }

    let mut copied = 0usize;
    for entry in fs::read_dir(&spirv_dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("spv") {
            let dst = out_spirv.join(path.file_name().unwrap());
            if !dst.exists() {
                fs::copy(&path, &dst).unwrap_or_else(|e| {
                    panic!("failed to copy {}: {e}", path.display())
                });
                copied += 1;
            }
        }
    }

    println!(
        "cargo:warning=vllm-vulkan: {} SPIR-V shaders ready ({} copied to OUT_DIR)",
        spv_count, copied
    );
}

// ─── macOS ────────────────────────────────────────────────────────────────────

fn link_macos() {
    // Probe standard locations for libvulkan.dylib:
    //   /opt/homebrew/lib — Homebrew vulkan-loader (CI / dev builds)
    //   /usr/local/lib    — KosmicKrisp end-user install (install.sh)
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
        // Attempt to link anyway; the linker error will be descriptive.
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }

    // Metal and related frameworks required by the Vulkan loader on macOS.
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

    let found = search_paths.iter().any(|dir| {
        Path::new(dir).join("libvulkan.so").exists()
    });

    if !found {
        println!(
            "cargo:warning=libvulkan.so not found. \
             Install libvulkan-dev: sudo apt-get install -y libvulkan-dev"
        );
    }

    // Add search paths so the linker finds libvulkan.so
    for dir in &search_paths {
        if Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    println!("cargo:rustc-link-lib=dylib=vulkan");
}
