use cmake;

fn main() {
    println!("cargo:rerun-if-changed=../deps/espeak-ng/src");
    println!("cargo:rustc-link-lib=static=espeak-ng");
    println!("cargo:rustc-link-lib=static=ucd");
    println!("cargo:rustc-link-lib=static=ucd");

    let build_dir = cmake::Config::new("../deps/espeak-ng")
        .configure_arg("-DUSE_ASYNC:BOOL=OFF")
        .configure_arg("-DUSE_MBROLA:BOOL=OFF")
        .configure_arg("-DUSE_LIBSONIC:BOOL=OFF")
        .configure_arg("-DUSE_LIBPCAUDIO:BOOL=OFF")
        .configure_arg("-DUSE_KLATT:BOOL=OFF")
        .configure_arg("-DUSE_SPEECHPLAYER:BOOL=OFF")
        .configure_arg("-DBUILD_SHARED_LIBS:BOOL=OFF")
        .build();

    println!(
        r"cargo:rustc-link-search={}",
        build_dir.join("lib").display()
    );
}
