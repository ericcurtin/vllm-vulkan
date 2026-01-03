//! Build script for vllm-vulkan
//!
//! This script compiles ggml with Vulkan backend support and generates
//! Rust FFI bindings using bindgen.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=ggml/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ggml_dir = PathBuf::from("ggml");

    // Check if ggml submodule exists
    if !ggml_dir.exists() {
        println!("cargo:warning=ggml submodule not found. Please run: git submodule update --init --recursive");
        // Create stub bindings for development without ggml
        generate_stub_bindings(&out_dir);
        return;
    }

    // Build ggml with Vulkan support using CMake
    let dst = cmake::Config::new(&ggml_dir)
        .define("GGML_VULKAN", "ON")
        .define("GGML_STATIC", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    // Link to ggml
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=ggml");

    // Link to Vulkan
    println!("cargo:rustc-link-lib=vulkan");

    // Generate bindings
    generate_ggml_bindings(&ggml_dir, &out_dir);
}

fn generate_ggml_bindings(ggml_dir: &PathBuf, out_dir: &PathBuf) {
    let header_path = ggml_dir.join("include").join("ggml.h");

    if !header_path.exists() {
        println!("cargo:warning=ggml.h not found, generating stub bindings");
        generate_stub_bindings(out_dir);
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .clang_arg(format!("-I{}/include", ggml_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Unable to generate ggml bindings");

    bindings
        .write_to_file(out_dir.join("ggml_bindings.rs"))
        .expect("Couldn't write ggml bindings!");
}

fn generate_stub_bindings(out_dir: &PathBuf) {
    // Generate stub bindings for development/CI without ggml
    let stub = r#"
//! Stub ggml bindings for development without ggml submodule
//! Run `git submodule update --init --recursive` to get the real bindings

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

// Stub types
pub type ggml_context = std::ffi::c_void;
pub type ggml_tensor = std::ffi::c_void;
pub type ggml_cgraph = std::ffi::c_void;
pub type ggml_backend_t = *mut std::ffi::c_void;
pub type ggml_backend_buffer_t = *mut std::ffi::c_void;
pub type ggml_backend_buffer_type_t = *mut std::ffi::c_void;

// Tensor types
pub const GGML_TYPE_F32: i32 = 0;
pub const GGML_TYPE_F16: i32 = 1;
pub const GGML_TYPE_Q4_0: i32 = 2;
pub const GGML_TYPE_Q4_1: i32 = 3;
pub const GGML_TYPE_Q5_0: i32 = 6;
pub const GGML_TYPE_Q5_1: i32 = 7;
pub const GGML_TYPE_Q8_0: i32 = 8;
pub const GGML_TYPE_Q8_1: i32 = 9;

// Backend types
pub const GGML_BACKEND_TYPE_CPU: i32 = 0;
pub const GGML_BACKEND_TYPE_GPU: i32 = 1;
pub const GGML_BACKEND_TYPE_GPU_SPLIT: i32 = 2;

// Maximum dimensions
pub const GGML_MAX_DIMS: usize = 4;
pub const GGML_MAX_NODES: usize = 16384;
"#;

    std::fs::write(out_dir.join("ggml_bindings.rs"), stub)
        .expect("Couldn't write stub bindings!");
}
