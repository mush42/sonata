use cmake;

fn main() {
    println!("cargo:rerun-if-changed=../../../deps/nanosnap/src");
    println!("cargo:rustc-link-lib=static=nanosnap");

    let build_dir = cmake::Config::new("../../../deps/nanosnap")
        .configure_arg("-DBUILD_SHARED_LIBS:BOOL=OFF")
        .build();

    println!(
        r"cargo:rustc-link-search={}",
        build_dir.join("lib").display()
    );
}
