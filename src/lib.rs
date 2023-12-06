//! Shared functionality between benchmarking examples.

use std::io::{Read, Seek, Write};
use std::num::NonZeroUsize;
use ndarray::{Array, Axis, Dim, s};
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_npy::{NpzReader, NpzWriter, ReadableElement, ReadNpzError, WritableElement, WriteNpzError};
use rand_distr::{Distribution, StandardNormal, Uniform};

/// Range of block sizes.
/// 
/// Block size can range from `min_size` up to and including
/// `max_size_inclusive`.
#[derive(Clone, Copy, Debug)]
pub struct BlockSizes {
    // Minimum block size
    min_size: NonZeroUsize,
    // Maximum block size
    max_size_inclusive: NonZeroUsize,
}
impl BlockSizes {
    /// Make a new `BlockSizes` ranging from `min_size` up to and inclusing
    /// `max_size_inclusive`. Returns None if the maximum size is less than the
    /// minimum size.
    pub fn new((min_size, max_size_inclusive): (NonZeroUsize, NonZeroUsize)) -> Option<Self> {
        if max_size_inclusive >= min_size {
            Some(Self {
                min_size,
                max_size_inclusive,
            })
        } else {
            None
        }
    }

    /// Delegates to [`BlockSizes::new()`]. In addition to requiring the maximum
    /// size to be greater than the minimum size, returns None if either size is
    /// zero.
    pub fn new_from_usize((min_size, max_size_inclusive): (usize, usize)) -> Option<Self> {
        let min_size = match NonZeroUsize::new(min_size) {
            Some(min_size) => min_size,
            None => { return None; },
        };
        let max_size_inclusive = match NonZeroUsize::new(max_size_inclusive) {
            Some(max_size_inclusive) => max_size_inclusive,
            None => { return None; },
        };
        Self::new((min_size, max_size_inclusive))
    }

    /// Gets the enclosed block sizes as a tuple of [`std::num::NonZeroUsize`]
    /// where the first element is the minimum size and the second element is
    /// the (inclusive) maximum size.
    pub fn get(&self) -> (NonZeroUsize, NonZeroUsize) {
        (self.min_size, self.max_size_inclusive)
    }

    /// Gets the enclosed minimum block size.
    pub fn min_size(&self) -> NonZeroUsize {
        self.min_size
    }

    /// Gets the enclosed maximum block size. The block size can be up to and
    /// _including_ this value.
    pub fn max_size_inclusive(&self) -> NonZeroUsize {
        self.max_size_inclusive
    }
}
impl Default for BlockSizes {
    fn default() -> Self {
        Self::new_from_usize((1, 8)).unwrap()
    }
}
impl std::fmt::Display for BlockSizes {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Block sizes from {} up to and including {}", self.min_size, self.max_size_inclusive)
    }
}

/// Parameters for generating mock data.
#[derive(Clone, Debug)]
pub struct MockParams {
    /// Number of observations
    pub n_obs: NonZeroUsize,
    /// Number of features/edges
    pub n_feat: NonZeroUsize,
    /// Number of predictors/covariates
    pub n_pred: NonZeroUsize,
    // Range of possible block/group sizes
    pub block_sizes: BlockSizes,
}
impl Default for MockParams {
    fn default() -> Self {
        Self {
            n_obs: NonZeroUsize::new(8192).unwrap(),
            n_feat: NonZeroUsize::new(((333 * 333) - 333) / 2).unwrap(),
            n_pred: NonZeroUsize::new(8).unwrap(),
            block_sizes: BlockSizes::default(),
        }
    }
}
impl std::fmt::Display for MockParams {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Mock data parameters:")?;
        writeln!(f, "Number of observations: {}", self.n_obs)?;
        writeln!(f, "Number of features: {}", self.n_feat)?;
        writeln!(f, "Number of predictors: {}", self.n_pred)?;
        writeln!(f, "{}", self.block_sizes)?;
        Ok(())
    }
}

/// Mock data of numeric type S, e.g. f64.
pub struct MockData<S: Clone> {
    /// Total number of blocks
    pub n_blocks: NonZeroUsize,
    /// 1-dimensional vector of block IDs. Each block ID is an integer ranging
    /// from 0 up to but excluding `n_blocks`.
    pub block_ids: Array<usize, Dim<[usize; 1]>>,
    /// Observation x features matrix of residuals
    pub resid: Array<S, Dim<[usize; 2]>>,
    /// Predictors x observations pseudoinverse of the design matrix
    pub x_pinv: Array<S, Dim<[usize; 2]>>,
}
impl <S> MockData<S>
where
    S: Clone,
    StandardNormal: Distribution<S>,
{
    /// Randomly generate mock data from mock parameters
    pub fn from_params(mock_params: MockParams) -> Self {
        // Rename mock_params to something shorter.
        let mp = mock_params;

        // Initialize a random number generator.
        let mut rng = rand::thread_rng();

        // Simulate an observations x features matrix of residuals from the
        // standard normal distribution.
        let resid = Array::<S, _>::random_using((mp.n_obs.get(), mp.n_feat.get()), StandardNormal, &mut rng);

        // Simulate a predictors x observations matrix to stand in for the
        // pseudoinverse of the design matrix, X.
        let x_pinv = Array::<S, _>::random_using((mp.n_pred.get(), mp.n_obs.get()), StandardNormal, &mut rng);
        
        // Simulate a vector of block ids. Each observation is assigned an integer
        // id from zero to n_blocks. We deliberately simulate non-continuous blocks
        // to benchmark the effect of cache misses in real data.
        let (block_ids, n_blocks) = {
            // Sample block sizes from a uniform distribution.
            let block_size = Uniform::new_inclusive(mp.block_sizes.min_size().get(), mp.block_sizes.max_size_inclusive().get());
            // Initialize a zero vector of block ids.
            let mut block_ids = Array::zeros(mp.n_obs.get());
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

        Self {
            n_blocks: NonZeroUsize::new(n_blocks).unwrap(),
            block_ids,
            x_pinv,
            resid,
        }
    }
}
impl <S> MockData<S>
where
    S: Clone + WritableElement,
{
    /// Save to a file.
    /// 
    /// Example:
    /// ```no_run
    /// use std::fs::File;
    /// let mock_data = MockData::from_params(MockParams::default());
    /// mock_data.save_npz_file(File::create("mock-data.npz")?)?;
    /// ```
    pub fn save_npz_file<W: Seek + Write>(&self, w: W) -> Result<(), WriteNpzError> {
        // Destructure self to make sure we handle all fields.
        let Self {
            n_blocks,
            block_ids,
            x_pinv,
            resid,
        } = self;

        // Convert usize to u64.
        let block_ids = block_ids.mapv(|x| x as u64);

        // Put number of blocks in an array.
        let n_blocks = Array::<u64,_>::from_elem((1,), n_blocks.get() as u64);

        // Initialize npz file.
        let mut npz = NpzWriter::new(w);

        // Add data to file.
        npz.add_array("n_blocks", &n_blocks)?;
        npz.add_array("block_ids", &block_ids)?;
        npz.add_array("x_pinv", x_pinv)?;
        npz.add_array("resid", resid)?;

        // Finish zip headers and flush file.
        npz.finish()?;

        // All done.
        Ok(())
    }
}
impl <S> MockData<S>
where
    S: Clone + ReadableElement,
{
    /// Load from a file.
    /// 
    /// Warning: Does not validate data in file and may panic if data is
    /// invalid.
    /// 
    /// Example:
    /// ```no_run
    /// use std::fs::File;
    /// let mock_data = MockData::from_npz_file(File::open("mock-data.npz")?)?;
    /// ```
    pub fn from_npz_file<R: Seek + Read>(r: R) -> Result<Self, ReadNpzError> {
        // Read the header of the npz file.
        let mut npz = NpzReader::new(r)?;

        // Read data from file.
        let n_blocks: Array<u64, Dim<[usize; 1]>> = npz.by_name("n_blocks")?;
        let block_ids: Array<u64, Dim<[usize; 1]>> = npz.by_name("block_ids")?;
        let x_pinv: Array<S, Dim<[usize; 2]>> = npz.by_name("x_pinv")?;
        let resid: Array<S, Dim<[usize; 2]>> = npz.by_name("resid")?;

        // Convert u64 to usize.
        let block_ids = block_ids.mapv_into_any(|x: u64| x as usize);

        // Extract number of blocks from array.
        let n_blocks = NonZeroUsize::new(n_blocks[0 as usize] as usize).unwrap(); // TODO error handling

        // Construct self.
        Ok(Self{
            n_blocks,
            block_ids,
            x_pinv,
            resid,
        })
    }
}
impl <S> MockData<S>
where
    S: Clone,
{
    /// Number of features.
    pub fn n_feat(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.resid.shape()[1]).unwrap()
    }

    /// Number of predictors.
    pub fn n_pred(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.x_pinv.shape()[0]).unwrap()
    }

    /// Number of observations.
    pub fn n_obs(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.resid.shape()[0]).unwrap()
    }
}

impl <S: Clone> std::fmt::Display for MockData<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Mock data parameters:")?;
        writeln!(f, "Number of observations: {}", self.n_obs())?;
        writeln!(f, "Number of features: {}", self.n_feat())?;
        writeln!(f, "Number of predictors: {}", self.n_pred())?;
        writeln!(f, "Number of blocks: {}", self.n_blocks)?;
        Ok(())
    }
}