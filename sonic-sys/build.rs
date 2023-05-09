extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=static=libsonic");
    println!("cargo:rerun-if-changed=../deps/sonic/sonic.h");

    cc::Build::new()
        .file("../deps/sonic/sonic.c")
        .include("../deps/sonic/sonic.h")
        .compile("libsonic");

    let bindings = bindgen::Builder::default()
        .header("../deps/sonic/sonic.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
