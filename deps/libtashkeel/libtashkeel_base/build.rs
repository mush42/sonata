fn main() {
    #[cfg(all(feature = "ort", feature = "tract"))]
    compile_error!("feature \"ort\" and feature \"tract\" cannot be enabled at the same time");
}
