//! Benchmark parallel implementation of SwE using ndarray and rayon threadpool.

// Force linking against blas and lapack backends.
extern crate blas_src;
extern crate lapack_src;

use ndarray::{s, Array, Axis, Dim, NewAxis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use rand_distr::{Distribution, StandardNormal, Uniform};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
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
    println!("Simulated {} blocks.", n_blocks);

    // Spin up a thread pool. //

    // Decide how many cpu cores to assign to the inner and outer thread pools.
    // The more cpus we use on the outer loop the more memory we will need.
    let ncpus = num_cpus::get();
    let ncpus_outer = if ncpus < 5 { 1 } else { 2 };
    let ncpus_inner = std::cmp::max(1, ncpus - ncpus_outer);

    // Create thread pools.
    let pool_outer = rayon::ThreadPoolBuilder::default()
        .num_threads(ncpus_outer)
        .build()?;
    let pool_inner = rayon::ThreadPoolBuilder::default()
        .num_threads(ncpus_inner)
        .build()?;
    println!("Outer thread pool has {} cpus.", ncpus_outer);
    println!("Inner thread pool has {} cpus.", ncpus_inner);

    // Compute the variance-covariance matrix of the regression coefficients, //
    // B, using the sandwhich estimator. //

    // Start the clock for benchmarking.
    let time = std::time::Instant::now();

    // Repeatedly compute cov_b n_rep times.
    let mut cov_b = Array::<f64, _>::zeros((n_pred, n_pred, n_feat));
    for rep in 0..n_rep {
        // Progress indicator.
        print!("Computing SwE, repetition {} of {}... ", rep + 1, n_rep);
        std::io::stdout().flush().unwrap();

        // Create a pair of channels to synchronize access to cov_b. We will
        // send 2-dimensional arrays of f64 over the channel along with a
        // feature index.
        let (tx, rx) = std::sync::mpsc::sync_channel::<(Array<f64, Dim<[usize; 2]>>, usize)>(
            ncpus_inner + ncpus_outer,
        );

        // Spawn a scoped thread to receive updates to cov_b.
        std::thread::scope(|s| {
            s.spawn(|| {
                // Initialize progress indicator.
                let mut n = 1; // number of updates
                let n_total = n_feat * n_blocks; // total number of updates
                let mut pct = 0; // percentage
                let msg = "0%"; // progress message to print
                print!("{}", msg);
                std::io::stdout().flush().unwrap();
                let mut msg_count = msg.as_bytes().len(); // bytes printed

                // Iterate over all updates to cov_b.
                let mut rx_iter = rx.into_iter();
                while let Some((cov_b_update, feat)) = rx_iter.next() {
                    // Update progress indicator every 1%.
                    let pct_new = 100. * (n as f32) / (n_total as f32);
                    if pct_new > (pct as f32) {
                        // Round to nearest whole percent.
                        pct = pct_new.round() as u8;
                        // Erase previous percentage with \x08 backspaces.
                        print!("{:\x08<1$}", "", msg_count);
                        // Print new percentage.
                        let msg = format!("{}%", pct);
                        print!("{}", msg);
                        msg_count = msg.as_bytes().len();
                        std::io::stdout().flush().unwrap();
                        // Don't update indicator until next whole percent.
                        pct += 1;
                    }
                    // Increment number of updates.
                    n = n + 1;

                    // Add update to the slice in cov_b indexed by its feature.
                    let mut slice = cov_b.slice_mut(s![.., .., feat]);
                    slice += &ndarray::ArrayView::from(&cov_b_update);
                }

                // Clear progress indicator.
                print!("{:\x08<1$}", "", msg_count);
            });

            // Enter the outer thread pool.
            pool_outer.install(|| {
                // Iterate over blocks.
                std::ops::Range {
                    start: 0,
                    end: n_blocks + 1, // Rust ranges exclude the last element
                }
                .into_par_iter()
                .for_each(|block_id| {
                    // Find indices for observations in this block.
                    // This is an opportunity for optimization, see https://github.com/rust-ndarray/ndarray/issues/466
                    // However, this would require major changes to ndarray :-(
                    let block_indices: Vec<_> = block_ids
                        .indexed_iter()
                        .filter_map(
                            |(index, &item)| if item == block_id { Some(index) } else { None },
                        )
                        .collect();

                    // Compute the half sandwich for all features in this block.
                    let half_sandwich = x_pinv
                        .select(Axis(1), &block_indices)
                        .dot(&resid.select(Axis(0), &block_indices));

                    // Enter the inner thread pool.
                    pool_inner.install(|| {
                        // Iterate over the features in half_sandwich.
                        half_sandwich
                            // Iterate over axis 2 (3rd dimension) of cov_b...
                            .axis_iter(Axis(1))
                            .into_par_iter()
                            // Keep track of the feature index.
                            .enumerate()
                            // Put a floor under the chunk size so that rayon does
                            // not overflow the stack by dividing the features up
                            // into too many teeny tiny chunks.
                            .with_min_len(half_sandwich.len_of(Axis(1)) / (ncpus_inner - 1))
                            .for_each(|(feat, half_sandwich)| {
                                // Compute the contribution to cov_b from this feature.
                                // TODO optimize by calling dsyrk directly through lax.
                                let half_sandwich = half_sandwich.slice(s![.., NewAxis]);
                                let tx = tx.clone();
                                tx.send((half_sandwich.dot(&half_sandwich.t()), feat))
                                    .unwrap(); // panic if receiver has hung up
                            });
                    })
                });
            });

            // Drop the last remaining handle to the transmit channel, signaling
            // that there are no more updates.
            drop(tx);
        });

        // Progress indicator.
        println!("done.");
    }

    // Print the time elapsed.
    let time_elapsed = time.elapsed();
    println!("Time elapsed: {:?}", time_elapsed);
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
