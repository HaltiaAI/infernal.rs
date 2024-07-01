use std::{
    env,
    path::{Path, PathBuf},
};

const GGML_METAL_METAL_PATH: &str = "llama.cpp/ggml-metal.metal";
const GGML_METAL_PATH: &str = "llama.cpp/ggml-metal.m";
const LLAMA_H_PATH: &str = "llama.cpp/llama.h";
const LLAMA_CPP_PATH: &str = "llama.cpp/llama.cpp";
const PATCHED_SRC_DIR: &str = "patched_src";
const PATCHED_INCLUDES_DIR: &str = "patched_includes";

fn main() {
    verify_state();

    println!("cargo:rerun-if-changed=llama.cpp");

    let out_dir = get_out_dir();

    build_ggml(&out_dir);
    build_llamacpp(&out_dir);

    eprintln!();

    bindgen_ggml();
    bindgen_llamacpp(&out_dir);
}

fn get_out_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is not defined"))
}

fn build_ggml(out_dir: &Path) {
    // ggml build

    let mut build = cc::Build::new();
    build
        .files([
            "llama.cpp/ggml.c",
            "llama.cpp/ggml-alloc.c",
            "llama.cpp/k_quants.c",
        ])
        // TODO: turn this on once we need k-quants
        .define("GGML_USE_K_QUANTS", None)
        .includes(["llama.cpp"]);
    if dbg!(cfg_cublas()) && !dbg!(cfg!(target_os = "macos")) {
        enable_cublas(&mut build, out_dir);
    } else if dbg!(cfg_clblast()) {
        enable_clblast(&mut build);
    } else if cfg!(any(target_os = "macos", target_os = "ios")) {
        if cfg_metal() {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
            println!("cargo:rerun-if-changed={GGML_METAL_METAL_PATH}");
            println!("cargo:rerun-if-changed={GGML_METAL_PATH}");
            patch_ggml_metal(&mut build, out_dir);
        } else {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            build.define("GGML_USE_ACCELERATE", None);
        }
    }

    compile(&mut build, "ggml");
}

fn build_llamacpp(out_dir: &Path) {
    let mut build = cc::Build::new();

    // Create custom include directory that does not include llama.cpp and llama.h as we are overwriting both files.
    let includes_dir = get_patched_includes_dir(out_dir);
    if !includes_dir.exists() {
        std::fs::create_dir_all(&includes_dir).unwrap_or_else(|e| {
            panic!("Could not create temporary include directory {includes_dir:?}: {e}")
        });
    }
    for entry in std::fs::read_dir("llama.cpp").unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let filename = path.file_name().unwrap().to_str().unwrap();
            if filename != "llama.cpp" && filename != "llama.h" {
                let dest = includes_dir.join(filename);
                std::fs::copy(&path, &dest)
                    .unwrap_or_else(|e| panic!("Could not copy {path:?} to {dest:?}: {e}"));
            }
        }
    }

    build
        .cpp(true)
        .flag("-std=c++11")
        // .file("llama.cpp/llama.cpp") NOTE: will be added later
        .includes([includes_dir]);
    // .includes(["llama.cpp"]);
    if cfg_cublas() && !cfg!(target_os = "macos") {
        build.define("GGML_USE_CUBLAS", "1");
    } else if cfg_clblast() {
        build.flag("-DGGML_USE_CLBLAST");
    } else if cfg!(any(target_os = "macos", target_os = "ios")) {
        if cfg_metal() {
            build.define("GGML_USE_METAL", "1");
        } else {
            build.define("GGML_USE_ACCELERATE", None);
        }
    }

    // this is disabled for now as we're not using it
    patch_llamacpp(&mut build, out_dir);
    compile(&mut build, "llama");
}

fn compile(build: &mut cc::Build, out: &str) {
    let compiler = build.get_compiler();

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let is_release = env::var("PROFILE").unwrap() == "release";

    // TODO: should we make it dependent on release builds?
    build.flag("-Ofast");

    let arch = target_arch.as_str();
    match arch {
        "aarch64" => {
            if compiler.is_like_clang() || compiler.is_like_gnu() {
                if env::var("HOST") == env::var("TARGET") {
                    build.flag("-mcpu=native");
                    build.flag("-mfpu=neon");
                } else {
                    #[allow(clippy::single_match)]
                    match target_os.as_str() {
                        "macos" | "ios" => {
                            build.flag("-mcpu=apple-m1");
                        }
                        _ => {}
                    }
                }
                build.flag("-pthread");
            } else {
                panic!("unsupported compiler");
            }
        }
        "x86" | "x86_64" => {
            let features = x86::Features::get();

            if compiler.is_like_clang() || compiler.is_like_gnu() {
                build.flag("-pthread");
                build.flag("-v");
                build.flag("-D_GNU_SOURCE");

                if features.avx {
                    build.flag("-mavx");
                }
                if features.avx2 {
                    build.flag("-mavx2");
                }
                if features.fma {
                    build.flag("-mfma");
                }
                if features.f16c {
                    build.flag("-mf16c");
                }
                if features.sse3 {
                    build.flag("-msse3");
                }
            } else if compiler.is_like_msvc() {
                match (features.avx2, features.avx) {
                    (true, _) => {
                        build.flag("/arch:AVX2");
                    }
                    (_, true) => {
                        build.flag("/arch:AVX");
                    }
                    _ => {}
                }
            }
        }
        _ => panic!("unsupported architecture: {arch}"),
    }

    if is_release {
        build.define("GGML_METAL_NDEBUG", "1");
    }

    build.warnings(false);

    if let Err(error) = build.try_compile(out) {
        eprintln!("Build error: {error}");

        std::process::exit(1);
    }
}

/// Verify the state of the repo to catch common newbie mistakes.
fn verify_state() {
    assert!(
        Path::new("llama.cpp/llama.cpp").exists(),
        "Could not find llama.cpp/llama.cpp. Try running `git submodule update --init`"
    );
}

fn patch_ggml_metal(build: &mut cc::Build, out_dir: &Path) {
    eprintln!("patching metal"); // TODO remove

    // HACK: patch ggml-metal.m so that it includes ggml-metal.metal, so that
    // a runtime dependency is not necessary
    let ggml_metal_path = patch_file(
        &get_patched_src_dir(out_dir),
        Path::new(GGML_METAL_PATH),
        |ggml_metal| {
            let ggml_metal_metal = std::fs::read_to_string(GGML_METAL_METAL_PATH)
                .expect("Could not read ggml-metal.metal")
                .replace('\\', "\\\\")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\"', "\\\"");

            // Patch the runtime metal shader file read with a compile-time string:

            let needle = r#"NSString * src  = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];"#;
            if !ggml_metal.contains(needle) {
                panic!("ggml-metal.m does not contain the needle to be replaced; the patching logic needs to be reinvestigated. Contact a `llm` developer!");
            }

            // Replace the runtime read of the file with a compile-time string
            let ggml_metal = ggml_metal.replace(
                needle,
                &format!(r#"NSString * src  = @"{ggml_metal_metal}";"#),
            );

            // Replace the judicious use of `fprintf` with the already-existing `metal_printf`,
            // backing up the definition of `metal_printf` first
            ggml_metal
                .replace(
                    r#"#define metal_printf(...) fprintf(stderr, __VA_ARGS__)"#,
                    "METAL_PRINTF_DEFINITION",
                )
                .replace("fprintf(stderr,", "metal_printf(")
                .replace(
                    "METAL_PRINTF_DEFINITION",
                    r#"#define metal_printf(...) fprintf(stderr, __VA_ARGS__)"#,
                )
        },
    );

    build.file(ggml_metal_path);
    build.flag("-DGGML_USE_METAL");

    #[cfg(not(debug_assertions))]
    build.flag("-DGGML_METAL_NDEBUG");
}

fn patch_llamacpp(build: &mut cc::Build, out_dir: &Path) {
    eprintln!("patching llama.cpp");

    let fn_decls = [
        "struct ggml_cgraph *infernal_ext_get_last_cgraph();",
        "void *infernal_ext_llama_mapping(struct llama_context *lctx);",
    ];
    let fn_defs = [
        "struct ggml_cgraph *infernal_ext_get_last_cgraph() { return LAST_CGRAPH; }",
        "void *infernal_ext_llama_mapping(struct llama_context *lctx) { return lctx->model.mapping->addr; }",
    ];

    // NB: include dir is added at the start of build_llamacpp
    patch_file(
        &get_patched_includes_dir(out_dir),
        Path::new(LLAMA_H_PATH),
        |mut llama_h| {
            let needle = "#ifdef __cplusplus";
            // Insert the function declearations just below the second cplusplus build switch.
            // First, find the position of the second cplusplus build switch.
            let pos = llama_h
                .match_indices(needle)
                .nth(1)
                .unwrap_or_else(|| panic!("{needle} must be present in llama.h"))
                .0;
            // Insert the functions just before the position.
            let mut s = fn_decls.join("\n");
            s += "\n";
            llama_h.insert_str(pos, &s);
            llama_h
        },
    );

    let llama_cpp_path = patch_file(
        &get_patched_src_dir(out_dir),
        Path::new(LLAMA_CPP_PATH),
        |mut llama_cpp| {
            // Insert the global graph pointer.
            let needle = "static struct ggml_cgraph * llama_build_graph(";
            let pos = llama_cpp
                .find(needle)
                .unwrap_or_else(|| panic!("{needle} must be present in llama.cpp"));
            llama_cpp.insert_str(pos, "static ggml_cgraph *LAST_CGRAPH = nullptr;\n");

            // Look for the place to insert the graph saving in. This is just before the `llama_build_graph`
            // function returns. So first find the function, then find the `return result`
            // statement, and inject the code before the return statement.
            // NB: Need to search again because the above insert changed the position.
            let pos = llama_cpp
                .find(needle)
                .unwrap_or_else(|| panic!("{needle} must be present in llama.cpp"));
            // Now search for the return statement from `pos` onwards.
            let offs = llama_cpp[pos..]
                .find("return result;")
                .unwrap_or_else(|| panic!("return result; must be present in llama.cpp"));
            llama_cpp.insert_str(pos + offs, "LAST_CGRAPH = result;\n");

            // Insert the functions at the end of the file.
            llama_cpp += &fn_defs.join("\n");

            llama_cpp
        },
    );

    build.file(llama_cpp_path);
}

fn patch_file(out_dir: &Path, path: &Path, mut f: impl FnMut(String) -> String) -> PathBuf {
    let mut s = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Could not read {path:?}"));
    s = f(s);

    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir).unwrap_or_else(|e| {
            panic!("Could not create temporary patched directory {out_dir:?}: {e}")
        });
    }

    let filename = path.file_name().unwrap().to_str().unwrap();
    let out_path = out_dir.join(filename);

    std::fs::write(&out_path, s)
        .unwrap_or_else(|e| panic!("Could not write temporary patched {out_path:?}: {e}"));

    out_path
}

fn get_patched_src_dir(out_dir: &Path) -> PathBuf {
    out_dir.join(PATCHED_SRC_DIR)
}
fn get_patched_includes_dir(out_dir: &Path) -> PathBuf {
    out_dir.join(PATCHED_INCLUDES_DIR)
}

fn cfg_cublas() -> bool {
    !cfg!(target_os = "macos") && !cfg!(target_os = "ios") && cfg!(feature = "cublas")
}

fn cfg_clblast() -> bool {
    !cfg!(target_os = "macos") && !cfg!(target_os = "ios") && cfg!(feature = "clblast")
}

fn cfg_metal() -> bool {
    cfg!(feature = "metal")
}

fn cuda_include_path() -> String {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_path = PathBuf::from(cuda_path);

        if cfg!(windows) {
            cuda_path
        } else {
            cuda_path.join("targets").join("x86_64-linux")
        }
        .join("include")
    } else if let Ok(cuda_include_path) = env::var("CUDA_INCLUDE_PATH") {
        PathBuf::from(cuda_include_path)
    } else {
        PathBuf::from("/usr/include")
    }
    .to_str()
    .expect("Could not build CUDA include path")
    .to_string()
}

fn cuda_lib_path() -> String {
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let cuda_path = PathBuf::from(cuda_path);

        if cfg!(windows) {
            cuda_path.join("lib").join("x64")
        } else {
            cuda_path.join("targets").join("lib")
        }
    } else if let Ok(cuda_lib_path) = env::var("CUDA_LIB_PATH") {
        PathBuf::from(cuda_lib_path)
    } else {
        PathBuf::from("/usr/lib/x86_64-linux-gnu")
    }
    .to_str()
    .expect("Could not build CUDA lib path")
    .to_string()
}

fn include_path(prefix: &str) -> String {
    if let Ok(path) = env::var(format!("{prefix}_PATH")) {
        PathBuf::from(path).join("include")
    } else if let Ok(include_path) = env::var(format!("{prefix}_INCLUDE_PATH")) {
        PathBuf::from(include_path)
    } else {
        PathBuf::from("/usr/include")
    }
    .to_str()
    .unwrap_or_else(|| panic!("Could not build {prefix} include path"))
    .to_string()
}

fn lib_path(prefix: &str) -> String {
    if let Ok(path) = env::var(format!("{prefix}_PATH")) {
        PathBuf::from(path).join("lib")
    } else if let Ok(lib_path) = env::var(format!("{prefix}_LIB_PATH")) {
        PathBuf::from(lib_path)
    } else {
        PathBuf::from("/usr/lib")
    }
    .to_str()
    .unwrap_or_else(|| panic!("Could not build {prefix} lib path"))
    .to_string()
}

fn enable_clblast(build: &mut cc::Build) {
    println!("cargo:rustc-link-lib=clblast");
    println!("cargo:rustc-link-lib=OpenCL");

    if cfg!(target_os = "linux") {
        //enable dynamic linking against stdc++
        println!(r"cargo:rustc-link-lib=dylib=stdc++");
    }

    build.file("llama.cpp/ggml-opencl.cpp");
    build.flag("-DGGML_USE_CLBLAST");

    let clblast_include_path = include_path("CLBLAST");
    let opencl_include_path = include_path("OPENCL");
    let clblast_lib_path = lib_path("CLBLAST");
    let opencl_lib_path = lib_path("OPENCL");

    if cfg!(windows) {
        build.flag("/MT");
    }

    println!(r"cargo:rustc-link-search=native={clblast_lib_path}");
    println!(r"cargo:rustc-link-search=native={opencl_lib_path}");

    build.flag(&format!(r"-I{clblast_include_path}"));
    build.flag(&format!(r"-I{opencl_include_path}"));
}

fn enable_cublas(build: &mut cc::Build, out_dir: &Path) {
    eprintln!("enabling cublas");

    let object_file = out_dir
        .join("llama.cpp")
        .join("ggml-cuda.o")
        .to_str()
        .expect("Could not build ggml-cuda.o filename")
        .to_string();

    let path = Path::new(&object_file);
    let parent_dir = path.parent().unwrap();

    std::fs::create_dir_all(parent_dir).unwrap();

    let include_path = cuda_include_path();
    let lib_path = cuda_lib_path();

    if cfg!(windows) {
        std::process::Command::new("nvcc")
            .arg("-ccbin")
            .arg(
                cc::Build::new()
                    .get_compiler()
                    .path()
                    .parent()
                    .unwrap()
                    .join("cl.exe"),
            )
            .arg("-I")
            .arg(&include_path)
            .arg("-o")
            .arg(&object_file)
            .arg("-x")
            .arg("cu")
            .arg("-maxrregcount=0")
            .arg("--machine")
            .arg("64")
            .arg("--compile")
            .arg("-cudart")
            .arg("static")
            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
            .arg("--generate-code=arch=compute_75,code=[compute_75,sm_75]")
            .arg("-D_WINDOWS")
            .arg("-DNDEBUG")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-D_CRT_SECURE_NO_WARNINGS")
            .arg("-D_MBCS")
            .arg("-DWIN32")
            .arg(r"-Illama.cpp\include\ggml")
            .arg(r"-Illama.cpp\include\ggml")
            .arg(r"llama.cpp\ggml-cuda.cu")
            .status()
            .unwrap();

        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");

        build.object(object_file);
        build.flag("-DGGML_USE_CUBLAS");
        build.include(&include_path);
    } else {
        std::process::Command::new("nvcc")
            .arg("--forward-unknown-to-host-compiler")
            .arg("-O3")
            .arg("-std=c++11")
            .arg("-fPIC")
            .arg("-Illama.cpp/include/ggml")
            .arg("-mtune=native")
            .arg("-pthread")
            .arg("--generate-code=arch=compute_52,code=[compute_52,sm_52]")
            .arg("--generate-code=arch=compute_61,code=[compute_61,sm_61]")
            .arg("--generate-code=arch=compute_75,code=[compute_75,sm_75]")
            .arg("-DGGML_USE_CUBLAS")
            .arg("-I/usr/local/cuda/include")
            .arg("-I/opt/cuda/include")
            .arg("-I")
            .arg(&include_path)
            .arg("-c")
            .arg("llama.cpp/ggml-cuda.cu")
            .arg("-o")
            .arg(&object_file)
            .status()
            .unwrap();

        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=culibos");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=dylib=stdc++");

        build.object(object_file);
        build.flag("-DGGML_USE_CUBLAS");
        build.include("/usr/local/cuda/include");
        build.include("/opt/cuda/include");
        build.include(include_path);
    }
}

fn get_supported_target_features() -> std::collections::HashSet<String> {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap()
        .split(',')
        .map(ToString::to_string)
        .collect()
}

mod x86 {
    #[allow(clippy::struct_excessive_bools)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Features {
        pub fma: bool,
        pub avx: bool,
        pub avx2: bool,
        pub f16c: bool,
        pub sse3: bool,
    }
    impl Features {
        pub fn get() -> Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::env::var("HOST") == std::env::var("TARGET") {
                return Self::get_host();
            }

            Self::get_target()
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub fn get_host() -> Self {
            Self {
                fma: std::is_x86_feature_detected!("fma"),
                avx: std::is_x86_feature_detected!("avx"),
                avx2: std::is_x86_feature_detected!("avx2"),
                f16c: std::is_x86_feature_detected!("f16c"),
                sse3: std::is_x86_feature_detected!("sse3"),
            }
        }

        pub fn get_target() -> Self {
            let features = crate::get_supported_target_features();
            Self {
                fma: features.contains("fma"),
                avx: features.contains("avx"),
                avx2: features.contains("avx2"),
                f16c: features.contains("f16c"),
                sse3: features.contains("sse3"),
            }
        }
    }
}

fn bindgen_llamacpp(out_dir: &Path) {
    let llama_bindings = bindgen::Builder::default()
        .header(
            get_patched_includes_dir(out_dir)
                .join("llama.h")
                .to_str()
                .unwrap(),
        )
        .allowlist_file(r".*llama.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    let llama_path = Path::new("./src/llama.rs");
    llama_bindings
        .write_to_file(llama_path)
        .expect("Couldn't create llama.rs module.");
}

fn bindgen_ggml() {
    let ggml_builder = bindgen::Builder::default()
        .header("llama.cpp/ggml.h")
        .header("llama.cpp/ggml-alloc.h")
        .allowlist_file(r".*ggml.h")
        .allowlist_file(r".*ggml-alloc.h");
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    let ggml_builder = ggml_builder
        .header("llama.cpp/ggml-metal.h")
        .allowlist_file(r".*ggml-metal.h");
    #[cfg(all(not(target_os = "macos"), feature = "cublas"))]
    let ggml_builder = ggml_builder
        .header("llama.cpp/ggml-cuda.h")
        .allowlist_file(r".*ggml-cuda.h");
    let ggml_bindings = ggml_builder
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    let ggml_path = Path::new("./src/ggml.rs");
    ggml_bindings
        .write_to_file(ggml_path)
        .expect("Couldn't create ggml.rs module.");
}
