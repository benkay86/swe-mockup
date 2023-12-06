//! Benchmark parallel implementation of SwE using ndarray and rayon threadpool.
//! 
//! This file benchmarks a single SwE computation. See [benchmark-perm.rs] for
//! benchmarking multiple SwE computations in parallel, as might occur in a wild
//! bootstrap.

// Force linking against blas and lapack backends.
extern crate blas_src;
extern crate lapack_src;

use ndarray::{s, Array, Axis, Dimension, NewAxis, ShapeBuilder};
use ndarray_rand::{RandomExt, SamplingStrategy};
use num_traits::Zero;
use rand_distr::{Distribution, StandardNormal, Uniform};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;
use std::io::Write; // for flushing stdout
use std::sync::{Condvar, Mutex};

// Structures for thread synchronization.

// Variables to be guarded by the mutex.
#[derive(Debug)]
struct CovBMutexInner<S, D>
where
    D: Dimension,
{
    // Number of threads running on the outer pool. Used to ensure there are not more
    // inner pools running than the number of cpus in the outer pool.
    outer_pool_reserved: usize,
    // Number of blocks being processed on the inner pool. Used to eliminate
    // lock contention on the inner thread pool.
    inner_pool_blocks: usize,
    // The variance-covariance matrix of b.
    cov_b: Array<S, D>,
}
impl<S, D> CovBMutexInner<S, D>
where
    S: Clone + Zero,
    D: Dimension,
{
    // Initialize `cov_b` to a matrix of zeros with shape `shape`.
    fn zeros<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            // Initially there are no inner pools reserved.
            outer_pool_reserved: 0,
            // Initially there are no inner pools running.
            inner_pool_blocks: 0,
            // Start with a matrix of zeros and add each block of the SwE.
            cov_b: Array::<S, D>::zeros(shape),
        }
    }
}

// Variables associated with cov_b and its condition variable.
// Needed for working around https://github.com/rayon-rs/rayon/issues/1105
#[derive(Debug)]
struct CovBCondvar<S, D>
where
    D: Dimension,
{
    // Mutex
    mutex: Mutex<CovBMutexInner<S, D>>,
    // Condition variable for `outer_pool_reserved`
    condvar_outer_reserved: Condvar,
    // Condition variable for `inner_pool_blocks`
    condvar_inner_blocks: Condvar,
}
impl<S, D> CovBCondvar<S, D>
where
    S: Clone + Zero,
    D: Dimension,
{
    // Initialize `mutex.cov_b` to a matrix of zeros with shape `shape`.
    fn zeros<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            mutex: Mutex::new(CovBMutexInner::zeros(shape)),
            condvar_outer_reserved: Condvar::new(),
            condvar_inner_blocks: Condvar::new(),
        }
    }
}

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
    // Because each inner thread pool blocks on a mutex, there is no performance
    // improvement from using more than two threads for the outer pool.
    let ncpus = std::thread::available_parallelism()?.get();
    let ncpus_outer = if ncpus < 5 { 1 } else { 2 };
    let ncpus_inner = std::cmp::max(1, ncpus - ncpus_outer);

    // Create thread pools.
    let pool_outer = ThreadPoolBuilder::default()
        // Make sure outer threads have large enough stacks (in bytes) to keep
        // track of forks/joins in the inner pool without overflowing.
        .stack_size(1024 * 1024 * ncpus_inner)
        .num_threads(ncpus_outer)
        .build()?;
    let pool_inner = ThreadPoolBuilder::default()
        .num_threads(ncpus_inner)
        .build()?;
    println!("Outer thread pool has {} cpus.", ncpus_outer);
    println!("Inner thread pool has {} cpus.", ncpus_inner);

    // Initialize cov_b to a matrix of zeros and set up some synchronization
    // primitives around it.
    let cov_b_condvar = CovBCondvar::<f64, _>::zeros((n_pred, n_pred, n_feat));

    // Compute the variance-covariance matrix of the regression coefficients, //
    // B, using the sandwhich estimator. //
    print!("Computing SwE... ");
    std::io::stdout().flush().unwrap();

    // Start the clock for benchmarking.
    let time = std::time::Instant::now();

    // Repeatedly compute cov_b n_rep times.
    for _ in 0..n_rep {
        // Reset cov_b to a matrix of zeros for this repetition.
        {
            cov_b_condvar.mutex.lock().unwrap().cov_b.fill(0.);
            // Lock is dropped and released here at end of scope.
        }

        // Enter the outer thread pool.
        pool_outer.install(|| {
            // Iterate over blocks.
            std::ops::Range {
                start: 0,
                end: n_blocks + 1, // Rust ranges exclude the last element
            }
            .into_par_iter()
            .for_each(|block_id| {
                // Reserve a thread on the outer pool to limit the number of
                // `half_sandwich` matrices computed in parallel to be not more
                // then the number of cpu resources on the outer pool. This is a
                // workaround for: https://github.com/rayon-rs/rayon/issues/1105
                // needed to prevent unbounded memory growth.
                // Note: unwrap() only panics if mutex is poisoned.
                {
                    // Wait until cpu resources are available.
                    let mut lock = cov_b_condvar.mutex.lock().unwrap();
                    while lock.outer_pool_reserved >= ncpus_outer {
                        lock = cov_b_condvar.condvar_outer_reserved.wait(lock).unwrap();
                    }
                    // Increment the number of inner pools reserved.
                    lock.outer_pool_reserved += 1;
                    // Lock is dropped and released here at end of scope.
                }

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

                // Don't send the block to the inner thread pool until the mutex
                // is available. Note: this will under-utilize the outer pool
                // if `ncpus_outer` is greater than 2.
                {
                    // Wait until no other blocks are running on the inner pool.
                    let mut lock = cov_b_condvar.mutex.lock().unwrap();
                    while lock.inner_pool_blocks > 0 {
                        lock = cov_b_condvar.condvar_inner_blocks.wait(lock).unwrap();
                    }
                    // Increment the number of blocks being processed on the
                    // inner pool.
                    lock.inner_pool_blocks += 1;
                    // Lock is dropped and released here at end of scope.
                }

                // Enter the inner thread pool.
                // Note: Per https://github.com/rust-ndarray/ndarray/issues/466
                // the call to `install()` may yield execution on _this_ thread
                // to another task.
                pool_inner.install(|| {
                    // Lock the mutex to get exclusive access to cov_b.
                    // Only panics if the mutex is poisoned.
                    let cov_b_mutex_inner = &mut (*cov_b_condvar.mutex.lock().unwrap());
                    let cov_b = &mut cov_b_mutex_inner.cov_b;

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

                    // Workaround for https://github.com/rayon-rs/rayon/issues/1105
                    // Decrement the number of blocks running on the inner pool.
                    cov_b_mutex_inner.inner_pool_blocks -= 1;
                    // Signal to a blocking thread that this inner pool is done.
                    cov_b_condvar.condvar_inner_blocks.notify_one();
                    // Decrement number of threads reserved on the outer pool.
                    cov_b_mutex_inner.outer_pool_reserved -= 1;
                    // Signal to a blocking thread that this inner pool is done.
                    cov_b_condvar.condvar_outer_reserved.notify_one();
                    // Lock is dropped and released here at end of scope.
                });
                // Note: Per https://github.com/rust-ndarray/ndarray/issues/466
                // don't block on a mutex here because `install()` may yield,
                // therefore we cannot guarantee when statements after
                // `install()` will execute or if they will deadlock.
            })
        })
    }

    // Print the time elapsed.
    let time_elapsed = time.elapsed();
    println!(" done.\nTime elapsed: {:?}", time_elapsed);
    println!(
        "That's {:?} per repetition.",
        time_elapsed / n_rep.try_into().unwrap()
    );

    // We're done multithreading; take cov_b out of the mutex.
    let cov_b = cov_b_condvar.mutex.into_inner().unwrap().cov_b; // panic if mutex is poisoned

    // Print an element of cov_b to make sure the optimizer sees we're using its
    // value and doesn't optimize away our benchmark!
    println!("cov_b[[0,0,0]] = {}", cov_b[[0, 0, 0]]);

    // All done, return success.
    Ok(())
}
