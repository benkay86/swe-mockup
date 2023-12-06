//! Benchmark parallel implementation of SwE using ndarray and rayon threadpool.
//! 
//! This algorithm is tailored for a parallel computation of multiple SwE as
//! might be used in a wild bootstrap. Edit the value of `n_rep` to control the
//! number of parallel, repeated computations of SwE. For an algorithm tailored
//! for a single SwE computation see (./benchmark-single.rs).

// Force linking against blas and lapack backends.
extern crate blas_src;
extern crate lapack_src;

use swe_mockup::{MockData, MockParams};

use ndarray::{s, Array, Axis, NewAxis};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io::Write; // for flushing stdout

#[allow(non_upper_case_globals)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmark of multiple, parallel SwE computations.");

    // Try to load mock data from file, otherwise generate it on the fly.
    let mock_data = if let Ok(file) = File::open("mock-data.npz") {
        print!("Reading mock data from mock-data.npz...");
        std::io::stdout().flush().unwrap();
        MockData::<f64>::from_npz_file(file)?
    } else {
        println!("File mock-data.npz not found.");
        println!("Consider running mock-npz to generate data.");
        print!("Generating mock data on the fly...");
        std::io::stdout().flush().unwrap();
        MockData::from_params(MockParams::default())
    };
    println!(" done.");
    print!("{}", mock_data);

    // Number of (non-parallel) repetitions of SwE comutation.
    let n_rep = 20;
    println!("Number of parallel repetitions: {}", n_rep);

    // Destructure mock data.
    let n_feat = mock_data.n_feat().get();
    let n_pred = mock_data.n_pred().get();
    let MockData {
        n_blocks,
        block_ids,
        resid,
        x_pinv
    } = mock_data;
    let n_blocks = n_blocks.get();

    // Spin up a thread pool. //
    let ncpus = std::thread::available_parallelism()?.get();
    ThreadPoolBuilder::default()
        // Make sure outer threads have large enough stacks (in bytes) to keep
        // track of forks/joins in the inner pool without overflowing.
        // .stack_size(1024 * 1024 * ncpus_inner)
        .num_threads(ncpus)
        .build_global()?;
    println!("Thread pool has {} cpus.", ncpus);

    // Compute the variance-covariance matrix of the regression coefficients, //
    // B, using the sandwhich estimator. //
    print!("Computing SwE... ");
    std::io::stdout().flush().unwrap();

    // Start the clock for benchmarking.
    let time = std::time::Instant::now();

    // Repeatedly compute cov_b n_rep times.
    (0..n_rep).into_par_iter().for_each(|_| {
        // Initialize an empty cov_b for this repetition.
        let mut cov_b = Array::<f64,_>::zeros((n_pred, n_pred, n_feat));

        // Iterate over blocks.
        // There is no performance benefit for doing this part in parallel.
        let blocks = std::ops::Range {
            start: 0,
            end: n_blocks + 1, // Rust ranges exclude the last element
        };
        for block in blocks {
            // Find indices for observations in this block.
            // This is an opportunity for optimization, see https://github.com/rust-ndarray/ndarray/issues/466
            // However, this would require major changes to ndarray :-(
            let block_indices: Vec<_> = block_ids
                .indexed_iter()
                .filter_map(|(index, &item)| if item == block { Some(index) } else { None })
                .collect();

            // Compute the half sandwich for all features in this block.
            let half_sandwich = x_pinv
                .select(Axis(1), &block_indices)
                .dot(&resid.select(Axis(0), &block_indices));

            // Iterate over the features in cov_b and half_sandwich
            // together. Zipping together the axis iterators proves to
            // the compiler that we will not go out of bounds,
            // eliminating the need for runtime bounds checking.
            cov_b
                // Iterate over axis 2 (3rd dimension) of cov_b...
                .axis_iter_mut(Axis(2))
                .into_par_iter()
                // ...together with axis 1 (columns) of half_sandwich.
                .zip(half_sandwich.axis_iter(Axis(1)))
                // Optionally put a floor under the chunk size so that
                // rayon does not overflow the stack by dividing the
                // features up into too many teeny tiny chunks.
                // .with_min_len(half_sandwich.len_of(Axis(1)) / (ncpus_inner + 1))
                .for_each(|(mut cov_b, half_sandwich)| {
                    // Compute the contribution to cov_b from this feature.
                    // TODO optimize by calling dsyrk directly through lax.
                    let half_sandwich = half_sandwich.slice(s![.., NewAxis]);
                    cov_b += &half_sandwich.dot(&half_sandwich.t());
                });
        }
    });

    // Print the time elapsed.
    let time_elapsed = time.elapsed();
    println!(" done.\nTime elapsed: {:?}", time_elapsed);
    println!(
        "That's {:?} per repetition.",
        time_elapsed / n_rep.try_into().unwrap()
    );

    // All done, return success.
    Ok(())
}
