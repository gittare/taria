// benchmarks/compress_bench.rs

use std::time::Instant;

fn main() {
    let src = "@gpu.kernel\ndef compress(x: Tensor) -> Latent:\n    return encoder(x)";
    let start = Instant::now();
    // TODO: Parse, lower, run on backend, measure time
    let duration = start.elapsed();
    println!("Compression pipeline took: {:?}", duration);
}
