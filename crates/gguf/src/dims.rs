use core::fmt;
use std::ops::Index;

/// The number of elements in a tensor, per dimension, up to 4 dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub dims: [usize; Self::MAX],
    n: usize,
}

pub const BATCH: usize = 3;
pub const DEPTH: usize = 2;
pub const ROW: usize = 1;
pub const COL: usize = 0;

impl Dims {
    pub const MAX: usize = 4;

    #[inline(always)]
    pub fn new_1d(cols: usize) -> Self {
        Self::new(&[cols])
    }

    #[inline(always)]
    pub fn new_2d(rows: usize, cols: usize) -> Self {
        Self::new(&[cols, rows])
    }

    #[inline(always)]
    pub fn new_3d(depth: usize, rows: usize, cols: usize) -> Self {
        Self::new(&[cols, rows, depth])
    }

    #[inline(always)]
    pub fn new_4d(batches: usize, depth: usize, rows: usize, cols: usize) -> Self {
        Self::new(&[cols, rows, depth, batches])
    }

    pub fn new(dims: &[usize]) -> Self {
        let mut this = Self::new_blank(dims.len());
        for (i, dim) in dims.iter().enumerate() {
            this.dims[i] = *dim;
        }
        this
    }

    pub fn new_blank(n: usize) -> Self {
        // TODO#perf: remove assert if showing up in profiles
        assert!(n <= Self::MAX);
        Self {
            // While we only use n_dims of this array, we still initialize everything to 1 in case
            // we need to pass it to ggml, which expects a 1 initialized array (1 being the identity
            // in multiplication, a property that ggml uses).
            dims: [1; Self::MAX],
            n,
        }
    }

    #[inline(always)]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline(always)]
    pub fn batches(&self) -> usize {
        self.dims[BATCH]
    }

    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.dims[DEPTH]
    }

    #[inline(always)]
    pub fn rows(&self) -> usize {
        self.dims[ROW]
    }

    #[inline(always)]
    pub fn cols(&self) -> usize {
        self.dims[COL]
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[usize] {
        &self.dims[..self.n]
    }

    #[doc(hidden)]
    pub fn as_array(&self) -> [usize; Self::MAX] {
        self.dims
    }

    #[inline(always)]
    pub fn set_dim(&mut self, i: usize, dim: usize) {
        self.dims[i] = dim;
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &usize> {
        self.dims.iter().take(self.n)
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// E.g. if this represents a 2-dimensional tensor, then dim0 * dim1 is going to be returned.
    #[inline(always)]
    pub fn total_elems(&self) -> usize {
        self.iter().product()
    }

    #[cfg(feature = "ggml")]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const usize {
        self.dims.as_ptr()
    }

    #[cfg(feature = "ggml")]
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut usize {
        self.dims.as_mut_ptr()
    }
}

impl Index<usize> for Dims {
    type Output = usize;

    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output {
        &self.dims[i]
    }
}

impl From<&'_ [usize]> for Dims {
    fn from(dims: &'_ [usize]) -> Self {
        Self::new(dims)
    }
}

impl<const N: usize> From<[usize; N]> for Dims {
    fn from(dims: [usize; N]) -> Self {
        Self::new(&dims)
    }
}

impl From<&'_ [u64]> for Dims {
    fn from(dims: &'_ [u64]) -> Self {
        assert!(dims.len() <= Self::MAX);
        let mut this = Self::new_blank(dims.len());
        for (i, dim) in dims.iter().enumerate() {
            this.dims[i] = *dim as usize;
        }
        this
    }
}

impl From<&'_ [u32]> for Dims {
    fn from(dims: &'_ [u32]) -> Self {
        assert!(dims.len() <= Self::MAX);
        let mut this = Self::new_blank(dims.len());
        for (i, dim) in dims.iter().enumerate() {
            this.dims[i] = *dim as usize;
        }
        this
    }
}

impl PartialEq<[usize]> for Dims {
    fn eq(&self, other: &[usize]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<&'_ [usize]> for Dims {
    fn eq(&self, other: &&'_ [usize]) -> bool {
        self.as_slice() == *other
    }
}

impl<const N: usize> PartialEq<[usize; N]> for Dims {
    fn eq(&self, other: &[usize; N]) -> bool {
        self.as_slice() == *other
    }
}

impl<const N: usize> PartialEq<&'_ [usize; N]> for Dims {
    fn eq(&self, other: &&'_ [usize; N]) -> bool {
        self.as_slice() == *other
    }
}

impl fmt::Display for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let skip = Self::MAX - self.n();
        for (i, dim) in self.dims.iter().rev().skip(skip).enumerate() {
            write!(f, "{dim}")?;
            if i < self.n() - 1 {
                write!(f, "x")?;
            }
        }
        Ok(())
    }
}
