// swamp-gpu/build.rs
// Build Mojo attention kernels into shared library for Rust FFI
// CPU target: generates AVX-512 via LLVM

use std::process::Command;
use std::path::Path;

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let kernel_path = Path::new(manifest_dir).join("kernels/attention.mojo");
    let so_path = Path::new(manifest_dir).join("libswamp_mojo.so");

    // Check if mojo is available
    let mojo_available = Command::new("which")
        .arg("mojo")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !mojo_available {
        println!("cargo:warning=mojo not found — Mojo attention kernel disabled");
        println!("cargo:warning=Will fall back to CPU attention in Rust");
        return;
    }

    // Compile Mojo → shared library (CPU target, AVX-512)
    println!("cargo:warning=mojo found — building CPU attention kernel");

    let status = Command::new("mojo")
        .args(&[
            "build",
            kernel_path.to_str().unwrap(),
            "-o",
            so_path.to_str().unwrap(),
            "--optimize", "fast",     // -O3 equivalent
            "--target", "cpu",         // CPU target (not CUDA)
            "--simd-width", "512",     // AVX-512 ZMM registers
            "--unroll-loops",          // aggressive unrolling
        ])
        .status()
        .expect("mojo build failed");

    if !status.success() {
        println!("cargo:warning=mojo build failed — falling back to CPU attention in Rust");
        return;
    }

    println!("cargo:warning=Mojo kernel built: {:?}", so_path);
    println!("cargo:rerun-if-changed=kernels/attention.mojo");
}
