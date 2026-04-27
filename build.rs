// SPDX-License-Identifier: Apache-2.0
//! Build script for vllm-vulkan.
//!
//! Platform routing:
//!   macOS (aarch64 / x86_64) — link against KosmicKrisp's MoltenVK-based
//!     Vulkan ICD. KosmicKrisp ships a `libvulkan.dylib` (or a symlink to
//!     `libMoltenVK.dylib`) that translates Vulkan calls to Metal.
//!
//!   Linux (x86_64 / aarch64) — link against the system `libvulkan.so`
//!     installed via the distro Vulkan loader (e.g. `libvulkan-dev` on
//!     Debian/Ubuntu).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

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
}

// ─── macOS ───────────────────────────────────────────────────────────────────

fn link_macos() {
    // KosmicKrisp installs its Vulkan loader as:
    //   /usr/local/lib/libvulkan.dylib  (Intel Homebrew prefix)
    //   /opt/homebrew/lib/libvulkan.dylib  (Apple-Silicon Homebrew prefix)
    //
    // The package is installable via:
    //   brew install KosmicKrisp/tap/kosmic-krisp
    // (or equivalently via MoltenVK: brew install molten-vk)
    //
    // We probe the two Homebrew prefixes first; if neither is found we fall
    // back to whatever the linker finds on the default search path (e.g. an
    // SDK-provided stub or a user-managed install).

    let homebrew_prefixes = ["/opt/homebrew", "/usr/local"];

    for prefix in &homebrew_prefixes {
        let lib_dir = format!("{prefix}/lib");
        if std::path::Path::new(&lib_dir).join("libvulkan.dylib").exists()
            || std::path::Path::new(&lib_dir).join("libMoltenVK.dylib").exists()
        {
            println!("cargo:rustc-link-search=native={lib_dir}");
            println!("cargo:rerun-if-changed={lib_dir}/libvulkan.dylib");
            break;
        }
    }

    // Prefer libvulkan (KosmicKrisp / MoltenVK loader); fall back to MoltenVK directly.
    println!("cargo:rustc-link-lib=dylib=vulkan");

    // Metal and IOKit are required by MoltenVK / KosmicKrisp.
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=IOKit");
    println!("cargo:rustc-link-lib=framework=IOSurface");
}

// ─── Linux ───────────────────────────────────────────────────────────────────

fn link_linux() {
    // Standard Vulkan loader; provided by libvulkan-dev / vulkan-loader.
    println!("cargo:rustc-link-lib=dylib=vulkan");
}
