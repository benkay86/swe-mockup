//! Benchmark parallel implementation of SwE using ndarray and rayon threadpool.

// Force linking against blas and lapack backends.
extern crate blas_src;
extern crate lapack_src;

use ndarray::{s, Array, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand_distr::{Distribution, StandardNormal, Uniform};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;
use std::io::Write; // for flushing stdout

#[allow(non_upper_case_globals)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Simulation parameters. //
    println!("Simulation parameters:");

    // Number of observations.
    const n_obs: usize = 8192;
    println!("Number of observations: {}", n_obs);

    // Number of features/edges.
    const n_feat: usize = ((333 * 333) - 333) / 2;
    println!("Number of features: {}", n_feat);

    // Number of predictors/covariates.
    const n_pred: usize = 8;
    println!("Number of predictors: {}", n_pred);

    // Range of possible block/group sizes from 1 to 8, inclusive.
    const min_block_size: usize = 1;
    const max_block_size: usize = 8;
    let block_size = Uniform::new_inclusive(min_block_size, max_block_size);
    println!("Minimum block size: {}", min_block_size);
    println!("Maximum block size: {}", max_block_size);

    // Number of repetitions/permutations.
    const n_rep: usize = 1;
    println!("Will repeat SwE calculation {} times.", n_rep);

    // Simulate mock-up data. //
    print!("Generating simulated data...");
    std::io::stdout().flush().unwrap();

    // Initialize a random number generator.
    let mut rng = rand::thread_rng();

    // Simulate an observations x features matrix of residuals from the standard
    // normal distribution.
    let resid = Array::<f64, _>::random_using((n_obs, n_feat), StandardNormal, &mut rng);

    // Simulate a predictors x observations matrix to stand in for the
    // pseudoinverse of the design matrix, X.
    let x_pinv = Array::<f64, _>::random_using((n_pred, n_obs), StandardNormal, &mut rng);

    // Simulate a vector of block ids. Each observation is assigned an integer
    // id from zero to n_blocks. We deliberately simulate non-continuous blocks
    // to benchmark the effect of cache misses in real data.
    let (block_ids, n_blocks) = {
        // Initialize a zero vector of block ids.
        let mut block_ids = Array::zeros(n_obs);
        // Initial conditions for loop.
        let mut block_id = 0;
        let mut block_start = 0;
        let mut block_end;
        // Loop, assigning ids block by block.
        while {
            // Random size for this block.
            block_end = block_start + block_size.sample(&mut rng);
            // When we reach the end of block_ids, let the last block default to
            // an id of zero.
            block_end <= block_ids.len_of(Axis(0))
        } {
            // Pre-increment block id.  The first id to be assigned will
            // therefore be one.
            block_id += 1;
            // Assign ids for this block.
            block_ids
                .slice_mut(s![block_start..block_end])
                .fill(block_id);
            // Next block starts where this block ends.
            block_start = block_end;
        }
        // Shuffle the order of the block ids.
        block_ids = block_ids.sample_axis_using(
            Axis(0),
            block_ids.len_of(Axis(0)),
            SamplingStrategy::WithoutReplacement,
            &mut rng,
        );
        // Return the vector of block ids and the last block id.
        (block_ids, block_id)
    };

    println!(" done.");

    // Spin up a thread pool. //

    // Decide how many cpu cores to assign to the inner and outer thread pools.
    // The more cpus we use on the outer loop the more memory we will need.
    let ncpus = num_cpus::get();
    let ncpus_outer = if ncpus < 5 { 1 } else { 2 };
    let ncpus_inner = std::cmp::max(1, ncpus - ncpus_outer);

    // Create thread pools.
    let pool_outer = ThreadPoolBuilder::default()
        .num_threads(ncpus_outer)
        .build()?;
    let pool_inner = ThreadPoolBuilder::default()
        .num_threads(ncpus_inner)
        .build()?;
    println!("Outer thread pool has {} cpus.", ncpus_outer);
    println!("Inner thread pool has {} cpus.", ncpus_inner);

    // Compute the variance-covariance matrix of the regression coefficients, //
    // B, using the sandwhich estimator. //
    print!("Computing SwE... ");
    std::io::stdout().flush().unwrap();

    // Start the clock for benchmarking.
    let time = std::time::Instant::now();

    // Repeatedly compute cov_b n_rep times.
    let mut cov_b = Array::<f64, _>::zeros((n_pred, n_pred, n_feat));
    for _ in 0..n_rep {
        // Assign cov_b the result from the outer thread pool.
        // Enter the outer thread pool.
        cov_b = pool_outer.install(|| {
            // Iterate over blocks.
            std::ops::Range {
                start: 0,
                end: n_blocks + 1, // Rust ranges exclude the last element
            }
            .into_par_iter()
            .map(|block_id| {
                // Find indices for observations in this block.
                // This is an opportunity for optimization, see https://github.com/rust-ndarray/ndarray/issues/466
                // However, this would require major changes to ndarray :-(
                let block_indices: Vec<_> = block_ids
                    .indexed_iter()
                    .filter_map(|(index, &item)| if item == block_id { Some(index) } else { None })
                    .collect();

                // Compute the half sandwich for all features in this block.
                let half_sandwich = x_pinv
                    .select(Axis(1), &block_indices)
                    .dot(&resid.select(Axis(0), &block_indices));

                // Enter the inner thread pool.
                pool_inner.install(|| {
                    // Allocate memory for cov_b for this block.
                    let mut cov_b = Array::<f64, _>::zeros((n_pred, n_pred, n_feat));

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
                        // Put a floor under the chunk size so that rayon does
                        // not overflow the stack by dividing the features up
                        // into too many teeny tiny chunks.
                        .with_min_len(half_sandwich.len_of(Axis(1)) / (ncpus_inner - 1))
                        .for_each(|(mut cov_b, half_sandwich)| {
                            // Compute the contribution to cov_b from this feature.
                            // TODO optimize by calling dsyrk directly through lax.
                            cov_b += half_sandwich.dot(&half_sandwich);
                        });

                    // Return cov_b for this block.
                    cov_b
                })
            })
            // Sum cov_b over blocks.
            .reduce(
                // Closure to return an "identity" cov_b for parallel summing,
                // in this case a matrix of zeros.
                || Array::<f64, _>::zeros((n_pred, n_pred, n_feat)),
                // Closure to perform the sum operation.
                |a, b| a + b,
            )
        });
    }

    // Print the time elapsed.
    let time_elapsed = time.elapsed();
    println!(" done.\nTime elapsed: {:?}", time_elapsed);
    println!(
        "That's {:?} per repetition.",
        time_elapsed / n_rep.try_into().unwrap()
    );

    // Print an element of cov_b to make sure the optimizer sees we're using its
    // value and doesn't optimize away our benchmark!
    println!("cov_b[[0,0,0]] = {}", cov_b[[0, 0, 0]]);

    // All done, return success.
    Ok(())
}
