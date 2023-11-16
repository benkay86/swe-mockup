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
Outer thread pool has 1 cpus.
Inner thread pool has 3 cpus.
Computing SwE...  done.
Time elapsed: 29.265938458s
That's 29.265938458s per repetition.
```

## See Also

See my old example repository for a Rosetta Stone of sorts for Matlab to Rust: https://github.com/benkay86/matlab-ndarray-tutorial