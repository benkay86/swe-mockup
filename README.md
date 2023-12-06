# Mockup for SwE Benchmark

Mockup of the sandwich estimator (SwE) calculation using simulated data for benchmarking purposes.

## Quick Start

[Install Rust](https://www.rust-lang.org/tools/install). On Linux this is a one-liner:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

> Hint: Later on you can update your Rust toolchain with [`rustup update`](https://rust-lang.github.io/rustup/basics.html).

Clone this repository and invoke [cargo](https://doc.rust-lang.org/cargo/), Rust's build tool.

```bash
git clone --recurse-submodules https://github.com/benkay86/swe-mockup.git
cd swe-mockup
cargo run --release
```

> Hint: Pull the latest changes to your local git repository with `git pull --recurse-submodules`.

> Hint: Use the `--release` profile to enable compiler performance enhancements.

Sample benchmark output:
```
Benchmark of multiple, parallel SwE computations.
File mock-data.npz not found.
Consider running mock-npz to generate data.
Generating mock data on the fly... done.
Mock data parameters:
Number of observations: 8192
Number of features: 55278
Number of predictors: 8
Number of blocks: 1800
Number of parallel repetitions: 20
Thread pool has 30 cpus.
Computing SwE...  done.
Time elapsed: 48.608350496s
That's 2.430417524s per repetition.
```

## Generating Mock Data

By default, the benchmarks will randomly generate a new set of mock data on each run. If you want the most consistent comparisons possible between benchmarks and runs, generate a single mock data set in advance and save it to disk. The data will be saved in the [NumPy format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) to `mock-data.npz` by default. The benchmarks will automatically use data from the `mock-data.npz` file if it is present (instead of generating new data).

```bash
cargo run --release --bin mock-npz # generate data
cargo run --release                # benchmark using generated data
```

To use the mock data in Matlab, use the [npy-matlab package](https://github.com/kwikteam/npy-matlab). This package does not support `*.npz` files, so first you will have to unzip the data and rename the data structures. Then you can call `readNPY()` in Matlab to load the files.

```bash
mkdir -p mock-data
unzip mock-data.npz -d mock-data
for FILE in mock-data/*; do mv ${FILE} ${FILE}.npy; done
```

## Selecting a Benchmark

The available benchmarks are listed under [src/bin](./src/bin). The default benchmark is a parallel computation of multiple sandwich estimator covariance matrices. To select a different benchmark, sich as a single computation of the SwE, specify the desired benchmark with `--bin`. For example:

```bash
cargo run --release --bin benchmark-single
```

## Matlab Benchmarks

To run the Matlab benchmarks, first generate some mock data and prepare it as above:

```bash
# Generate mock data
cargo run --release --bin mock-npz
# Prepare the data for Matlab
mkdir -p mock-data
unzip mock-data.npz -d mock-data
for FILE in mock-data/*; do mv ${FILE} ${FILE}.npy; done
```

Then enter the [`matlab`](./matlab) directory and run the desired benchmark.

```bash
cd matlab
matlab -nodesktop -r benchmark_multi
```

## Troubleshooting

If you get a compilation error to the effect of `cannot find -lopenblas` then you either do not have openblas installed on your system, or else it is not installed in a place where `openblas-src` and `blas-src` can find it. Either check to make sure openblas is installed correctly, or else edit [`Cargo.toml`](./Cargo.toml), replacing `"openblas-system"` and `"system"` with `"openblas-static"` and `"static"`. This will compile a bundled version of the openblas and lapack source and statically link to it. Note that this workaround will dramatically increase compilation size, increase the size of the binary, and potentialy build a less-highly-optimized version of openblas than the one bundled with your system.

## See Also

See my old example repository for a Rosetta Stone of sorts for Matlab to Rust: https://github.com/benkay86/matlab-ndarray-tutorial