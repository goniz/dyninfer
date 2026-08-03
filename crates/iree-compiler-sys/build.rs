//! Cargo-only bindgen path. Bazel uses `rust_bindgen` + `rustc_env` instead.

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("BAZEL_BINDGEN").ok().as_deref() == Some("1") {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let wrapper = manifest_dir
        .join("../../third_party/iree_compiler_c_api/wrapper.h")
        .canonicalize()
        .expect("wrapper.h (run from workspace crates/iree-compiler-sys)");
    let include = manifest_dir
        .join("../../third_party/iree_compiler_c_api/include")
        .canonicalize()
        .expect("include/");

    println!("cargo:rerun-if-changed={}", wrapper.display());
    println!(
        "cargo:rerun-if-changed={}",
        include.join("iree/compiler/embedding_api.h").display()
    );

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_function("ireeCompiler.*")
        .allowlist_type("iree_compiler_.*")
        .allowlist_var("IREE_COMPILER_.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .layout_tests(false)
        .ctypes_prefix("core::ffi")
        .use_core()
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen IREE compiler API");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out.join("iree_compiler_bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("write bindings");
    println!(
        "cargo:rustc-env=IREE_COMPILER_BINDINGS={}",
        bindings_path.display()
    );

    if let Ok(lib) = env::var("DYNINFER_IREE_COMPILER_LIB") {
        let lib = PathBuf::from(lib);
        if let Some(dir) = lib.parent() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    } else {
        let root = manifest_dir.join("../..");
        if let Ok(lib_root) = std::fs::read_dir(root.join("third_party/iree-venv/lib")) {
            for entry in lib_root.flatten() {
                let dir = entry.path().join("site-packages/iree/compiler/_mlir_libs");
                if dir.join("libIREECompiler.so").is_file() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
                    break;
                }
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=IREECompiler");
}
