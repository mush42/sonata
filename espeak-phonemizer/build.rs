use fs_extra::{copy_items, dir::CopyOptions};
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Link espeak-ng
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-lib=espeak-ng");
    #[cfg(not(target_os = "windows"))]
    println!("cargo:rustc-link-lib=libespeak-ng");


    #[cfg(target_os = "windows")]
    {
        let win_prebuilt_espeak_ng_directory = manifest_dir
            .parent()
            .unwrap()
            .join("deps")
            .join("windows")
            .join("espeak-ng-build");
        let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
        let es_prebuilt_dir = win_prebuilt_espeak_ng_directory.join(target_arch);
        let espeak_build_directory = fs::canonicalize(es_prebuilt_dir).unwrap();

        // Add espeak-ng.lib to linker search path
        let espeak_lib_directory = espeak_build_directory.join("lib");
        println!(
            "cargo:rustc-link-search=native={}",
            espeak_lib_directory.display()
        );

        // Copy espeak-ng.dll to out_dir
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_dir = out_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let target_file = target_dir.join("espeak-ng.dll");
        if !target_file.exists() {
            let source_file = espeak_build_directory.join("bin/espeak-ng.dll");
            fs::copy(source_file, target_file).unwrap();
        }

        // Copy espeak-ng data to out_dir
        let source_folder = espeak_build_directory
            .parent()
            .unwrap()
            .join("espeak-ng-data");
        if !target_dir.join("espeak-ng-data").exists() {
            copy_items(&[source_folder], target_dir, &CopyOptions::new()).unwrap();
        }
    }
}
