use git2::Repository;
use std::path::{Path, PathBuf};
use std::process::Command;

static LLVM_REPO: &str = "https://github.com/llvm/llvm-project";

fn shallow_clone(path: &Path, repo: &str) -> Repository {
    let mut child = Command::new("git")
        .args(&[
            "clone",
            "-b",
            "llvmorg-15.0.7",
            // "--depth",
            // "1",
            "--recurse-submodules",
            "--shallow-submodules",
            "-j10",
            repo,
            path.to_str().unwrap(),
        ])
        .spawn()
        .expect("failed to execute process");
    child.wait().unwrap();

    Repository::open(path).unwrap()
}

/// use cached repo if it exists, otherwise clone it
fn get_repo(path: &Path, repo: &str) -> Repository {
    println!("Checking for cached repo at: {}", path.to_str().unwrap());
    let out = if path.exists() {
        Repository::open(path).unwrap()
    } else {
        shallow_clone(path, repo)
    };

    out
}

fn build_mlir(llvm_proj_dir: &Path, out_dir: &Path) {
    let mlir_out_dir = out_dir.join("mlir-build");

    if !mlir_out_dir.exists() {
        std::fs::create_dir_all(&mlir_out_dir).unwrap();
    }

    let mut cfg = cxx_build::bridge("src/lib.rs");
    cfg
        .cpp(true)
        .include("mlir-build/build/include")
        .include("llvm-project/mlir/include")
        .include("llvm-project/llvm/include")
        .out_dir(".");

    cmake::Config::new(&llvm_proj_dir.join("llvm"))
        .out_dir(mlir_out_dir.clone())
        .define("CXX_STANDARD", "17")
        .define("LLVM_ENABLE_PROJECTS", "mlir")
        .define("LLVM_TARGETS_TO_BUILD", "AArch64;AMDGPU;host")
        .define("LLVM_BUILD_EXAMPLES", "ON")
        .define("MLIR_ENABLE_CUDA_RUNNER", "OFF")
        .target("check-mlir")
        .build_target("all")
        .init_c_cfg(cfg.clone())
        .init_cxx_cfg(cfg.clone())
        .build();
}

fn main() {
    let out_dir = PathBuf::from(".");
    let llvm_dir = out_dir.join("llvm-project");

    let repo = get_repo(&llvm_dir, &LLVM_REPO);

    build_mlir(&llvm_dir, &out_dir);


}
