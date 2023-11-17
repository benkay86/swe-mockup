# Mockup for SwE Benchmark

Mockup of the sandwich estimator (SwE) calculation using simulated data for benchmarking purposes.

## Quick Start

[Install Rust](https://www.rust-lang.org/tools/install). On Linux this rust is a one-liner:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

> Hint: Later on you can update your Rust toolchain with [`rustup update`](https://rust-lang.github.io/rustup/basics.html).

Clone this repository and invoke [cargo](https://doc.rust-lang.org/cargo/), Rust's build tool.

```bash
git clone https://github.com/benkay86/swe-mockup.git
cd swe-mockup
cargo run --release
```

> Hint: Use the `--release` profile to enable compiler performance enhancements.

Sample benchmark output:
```
Simulation parameters:
Number of observations: 8192
Number of features: 55278
Number of predictors: 8
Minimum block size: 1
Maximum block size: 8
Will repeat SwE calculation 1 times.
Generating simulated data... done.
Simulated 1878 blocks.
Outer thread pool has 2 cpus.
Inner thread pool has 14 cpus.
Computing SwE...  done.
Time elapsed: 23.726886259s
That's 23.726886259s per repetition.
```

## Troubleshooting

If you get a compilation error to the effect of `cannot find -lopenblas` then you either do not have openblas installed on your system, or else it is not installed in a place where `openblas-src` and `blas-src` can find it. Either check to make sure openblas is installed correctly, or else edit [`Cargo.toml`](./Cargo.toml), replacing `"openblas-system"` and `"system"` with `"openblas-static"` and `"static"`. This will compile a bundled version of the openblas and lapack source and statically link to it. Note that this workaround will dramatically increase compilation size, increase the size of the `benchmark` binary, and potentialy build a less-highly-optimized version of openblas than the one bundled with your system.

## See Also

See my old example repository for a Rosetta Stone of sorts for Matlab to Rust: https://github.com/benkay86/matlab-ndarray-tutorial