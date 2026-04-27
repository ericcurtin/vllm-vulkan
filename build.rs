// SPDX-License-Identifier: Apache-2.0
//! Build script for vllm-vulkan.
//!
//! Platform routing:
//!   macOS (aarch64 / x86_64) — link against libMoltenVK.dylib installed by
//!     `brew install molten-vk` (or KosmicKrisp). MoltenVK translates Vulkan
//!     calls to Metal at runtime. Note: Homebrew's molten-vk does NOT ship a
//!     libvulkan.dylib loader, so we link MoltenVK directly.
//!
//!   Linux (x86_64 / aarch64) — link against the system libvulkan.so loader
//!     installed via libvulkan-dev (Debian/Ubuntu) or vulkan-loader-devel
//!     (Fedora/RHEL).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match target_os.as_str() {
        "macos" => link_macos(),
        "linux" => link_linux(),
        other => {
            println!(
                "cargo:warning=vllm-vulkan: unsupported target OS '{other}'. \
                 Only macOS (via KosmicKrisp/MoltenVK) and Linux are supported."
            );
        }
    }
}

// ─── macOS ───────────────────────────────────────────────────────────────────

fn link_macos() {
    // Homebrew's molten-vk installs libMoltenVK.dylib (not libvulkan.dylib).
    // Probe both Homebrew prefixes, then link whichever library is present.
    let homebrew_prefixes = ["/opt/homebrew", "/usr/local"];

    let mut linked = false;
    for prefix in &homebrew_prefixes {
        let lib_dir = format!("{prefix}/lib");
        let has_moltenvk = std::path::Path::new(&lib_dir).join("libMoltenVK.dylib").exists();
        let has_vulkan   = std::path::Path::new(&lib_dir).join("libvulkan.dylib").exists();

        if has_moltenvk || has_vulkan {
            println!("cargo:rustc-link-search=native={lib_dir}");
            if has_moltenvk {
                println!("cargo:rustc-link-lib=dylib=MoltenVK");
            } else {
                println!("cargo:rustc-link-lib=dylib=vulkan");
            }
            linked = true;
            break;
        }
    }

    if !linked {
        println!(
            "cargo:warning=Neither libMoltenVK.dylib nor libvulkan.dylib found. \
             Install with: brew install molten-vk"
        );
        // Attempt to link anyway; the linker error will be descriptive.
        println!("cargo:rustc-link-lib=dylib=MoltenVK");
    }

    // Metal and related frameworks required by MoltenVK.
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}

// ─── Linux ───────────────────────────────────────────────────────────────────

fn link_linux() {
    // Standard Vulkan loader; provided by libvulkan-dev / vulkan-loader.
    //
    // Probe for the unversioned linker stub (libvulkan.so), not the runtime
    // library (libvulkan.so.1), because -lvulkan resolves via the unversioned
    // symlink which is only present in the -dev package.
    let search_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
    ];

    let found = search_paths.iter().any(|dir| {
        std::path::Path::new(dir).join("libvulkan.so").exists()
    });

    if !found {
        println!(
            "cargo:warning=libvulkan.so not found. \
             Install libvulkan-dev: sudo apt-get install -y libvulkan-dev"
        );
    }

    println!("cargo:rustc-link-lib=dylib=vulkan");
}
