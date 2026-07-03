// swamp-gpu/build.rs
// Build CUDA kernels if nvcc is available

use std::process::Command;

fn main() {
    // Check if nvcc is available
    let nvcc_available = Command::new("which")
        .arg("nvcc")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !nvcc_available {
        println!("cargo:warning=nvcc not found — GPU library will not be built");
        println!("cargo:warning=GPU acceleration will fall back to CPU");
        return;
    }

    println!("cargo:warning=nvcc found — building GPU library");

    let status = Command::new("make")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("make failed");

    if !status.success() {
        panic!("GPU library build failed (make returned {})", status);
    }

    // Tell cargo to re-run build.rs if the .cu file changes
    println!("cargo:rerun-if-changed=kernels/fused_attention.cu");
    println!("cargo:rerun-if-changed=Makefile");
}
