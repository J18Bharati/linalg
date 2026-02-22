use std::fmt;
use std::ops::{Index, IndexMut};

use crate::numeric::Numeric;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: Numeric, const RANK: usize> {
    pub(crate) shape: [usize; RANK],
    pub(crate) data: Vec<T>,
}

impl<T: Numeric, const RANK: usize> Tensor<T, RANK> {
    /// Total number of elements.
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    /// The shape array.
    #[inline]
    pub fn shape(&self) -> &[usize; RANK] {
        &self.shape
    }

    /// Alias for `num_elements`.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat slice of the underlying data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Convert a multi-dimensional index to a flat offset (row-major).
    fn flat_index(&self, idx: &[usize; RANK]) -> usize {
        let mut offset = 0;
        for (i, &ix) in idx.iter().enumerate() {
            assert!(
                ix < self.shape[i],
                "index {} out of bounds for axis {} with size {}",
                ix,
                i,
                self.shape[i],
            );
            offset = offset * self.shape[i] + ix;
        }
        offset
    }

    // -- Constructors --

    /// Tensor filled with zeros.
    pub fn zeros(shape: [usize; RANK]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: vec![T::zero(); n],
        }
    }

    /// Tensor filled with ones.
    pub fn ones(shape: [usize; RANK]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: vec![T::one(); n],
        }
    }

    /// Tensor filled with a given value.
    pub fn full(shape: [usize; RANK], value: T) -> Self {
        let n: usize = shape.iter().product();
        Self {
            shape,
            data: vec![value; n],
        }
    }

    /// Build a tensor from a shape and a flat `Vec`.
    ///
    /// # Panics
    /// Panics if `data.len()` doesn't match the product of `shape`.
    pub fn from_shape_vec(shape: [usize; RANK], data: Vec<T>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            n,
        );
        Self { shape, data }
    }

    /// Build a tensor by calling `f` with each multi-dimensional index.
    pub fn from_shape_fn(shape: [usize; RANK], f: impl Fn([usize; RANK]) -> T) -> Self {
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut idx = [0usize; RANK];

        for _ in 0..n {
            data.push(f(idx));
            // Increment the index in row-major order.
            for axis in (0..RANK).rev() {
                idx[axis] += 1;
                if idx[axis] < shape[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }

        Self { shape, data }
    }
}

// -- Indexing --

impl<T: Numeric, const RANK: usize> Index<[usize; RANK]> for Tensor<T, RANK> {
    type Output = T;

    #[inline]
    fn index(&self, idx: [usize; RANK]) -> &T {
        let i = self.flat_index(&idx);
        &self.data[i]
    }
}

impl<T: Numeric, const RANK: usize> IndexMut<[usize; RANK]> for Tensor<T, RANK> {
    #[inline]
    fn index_mut(&mut self, idx: [usize; RANK]) -> &mut T {
        let i = self.flat_index(&idx);
        &mut self.data[i]
    }
}

// -- Display --

impl<T: Numeric> fmt::Display for Tensor<T, 0> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data[0])
    }
}

impl<T: Numeric> fmt::Display for Tensor<T, 1> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}")?;
        }
        write!(f, "]")
    }
}

impl<T: Numeric> fmt::Display for Tensor<T, 2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = self.shape[0];
        let cols = self.shape[1];
        writeln!(f, "[")?;
        for r in 0..rows {
            write!(f, "  [")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[r * cols + c])?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

// Fallback for rank >= 3: just print shape + flat data.
macro_rules! impl_display_fallback {
    ($($R:literal),*) => {
        $(
            impl<T: Numeric> fmt::Display for Tensor<T, $R> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    write!(f, "Tensor(shape={:?}, data={:?})", self.shape, self.data)
                }
            }
        )*
    };
}

impl_display_fallback!(3, 4, 5, 6);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_ones() {
        let z = Tensor::<f64, 2>::zeros([2, 3]);
        assert_eq!(z.shape(), &[2, 3]);
        assert_eq!(z.len(), 6);
        assert!(z.as_slice().iter().all(|&v| v == 0.0));

        let o = Tensor::<f64, 1>::ones([4]);
        assert!(o.as_slice().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn from_shape_vec_and_index() {
        let t = Tensor::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(t[[0, 0]], 1);
        assert_eq!(t[[0, 2]], 3);
        assert_eq!(t[[1, 0]], 4);
        assert_eq!(t[[1, 2]], 6);
    }

    #[test]
    fn index_mut() {
        let mut t = Tensor::<i32, 1>::zeros([3]);
        t[[0]] = 10;
        t[[2]] = 30;
        assert_eq!(t.as_slice(), &[10, 0, 30]);
    }

    #[test]
    fn from_shape_fn() {
        let t = Tensor::from_shape_fn([2, 2], |[r, c]| (r * 2 + c) as f64);
        assert_eq!(t.as_slice(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "does not match shape")]
    fn from_shape_vec_bad_len() {
        let _ = Tensor::from_shape_vec([2, 3], vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_out_of_bounds() {
        let t = Tensor::<i32, 1>::zeros([3]);
        let _ = t[[5]];
    }

    #[test]
    fn display_rank0() {
        let s = Tensor::from_shape_vec([], vec![42.0_f64]);
        assert_eq!(format!("{s}"), "42");
    }

    #[test]
    fn display_rank1() {
        let v = Tensor::from_shape_vec([3], vec![1, 2, 3]);
        assert_eq!(format!("{v}"), "[1, 2, 3]");
    }
}
