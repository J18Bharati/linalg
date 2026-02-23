use crate::numeric::Numeric;
use crate::tensor::Tensor;

/// Methods specific to rank-2 tensors (matrices).
impl<T: Numeric> Tensor<T, 2> {
    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Create a matrix from a vector of row vectors.
    ///
    /// # Panics
    /// Panics if rows are empty or have inconsistent lengths.
    pub fn from_rows(rows: &[Vec<T>]) -> Self {
        assert!(!rows.is_empty(), "cannot create matrix from empty rows");
        let nrows = rows.len();
        let ncols = rows[0].len();
        assert!(ncols > 0, "rows must not be empty");
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                ncols,
                "row {} has length {}, expected {}",
                i,
                row.len(),
                ncols,
            );
        }
        let data: Vec<T> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Self {
            shape: [nrows, ncols],
            data,
        }
    }

    /// Build a matrix by calling `f(row, col)` for each element.
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self {
            shape: [rows, cols],
            data,
        }
    }

    /// Identity matrix of size `n x n`.
    pub fn identity(n: usize) -> Self {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        Self {
            shape: [n, n],
            data,
        }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Self {
        let (r, c) = (self.rows(), self.cols());
        let mut data = vec![T::zero(); r * c];
        for i in 0..r {
            for j in 0..c {
                data[j * r + i] = self.data[i * c + j];
            }
        }
        Self {
            shape: [c, r],
            data,
        }
    }

    /// Matrix multiplication (self * other).
    /// Uses i-k-j loop order for better cache performance.
    ///
    /// # Panics
    /// Panics if inner dimensions don't match.
    pub fn matmul(&self, other: &Self) -> Self {
        let (m, k1) = (self.rows(), self.cols());
        let (k2, n) = (other.rows(), other.cols());
        assert_eq!(k1, k2, "matmul: inner dimensions mismatch ({k1} vs {k2})");

        let mut data = vec![T::zero(); m * n];

        // i-k-j loop order
        for i in 0..m {
            for k in 0..k1 {
                let a_ik = self.data[i * k1 + k];
                for j in 0..n {
                    data[i * n + j] = data[i * n + j] + a_ik * other.data[k * n + j];
                }
            }
        }

        Self {
            shape: [m, n],
            data,
        }
    }

    /// Matrix-vector multiplication (self * v).
    ///
    /// # Panics
    /// Panics if the vector length doesn't match the number of columns.
    pub fn matvec(&self, v: &Tensor<T, 1>) -> Tensor<T, 1> {
        let (m, n) = (self.rows(), self.cols());
        assert_eq!(
            v.shape()[0],
            n,
            "matvec: vector length {} != matrix cols {}",
            v.shape()[0],
            n,
        );

        let data: Vec<T> = (0..m).map(|i| self.row(i).inner_product(v)).collect();

        Tensor {
            shape: [m],
            data,
        }
    }

    /// Extract row `i` as a vector.
    pub fn row(&self, i: usize) -> Tensor<T, 1> {
        assert!(i < self.rows(), "row index {i} out of bounds");
        let c = self.cols();
        let data = self.data[i * c..(i + 1) * c].to_vec();
        Tensor {
            shape: [c],
            data,
        }
    }

    /// Extract column `j` as a vector.
    pub fn col(&self, j: usize) -> Tensor<T, 1> {
        assert!(j < self.cols(), "col index {j} out of bounds");
        let (r, c) = (self.rows(), self.cols());
        let data: Vec<T> = (0..r).map(|i| self.data[i * c + j]).collect();
        Tensor {
            shape: [r],
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Matrix, Vector};

    #[test]
    fn identity() {
        let eye = Matrix::<f64>::identity(3);
        assert_eq!(eye.rows(), 3);
        assert_eq!(eye.cols(), 3);
        assert_eq!(eye[[0, 0]], 1.0);
        assert_eq!(eye[[0, 1]], 0.0);
        assert_eq!(eye[[1, 1]], 1.0);
    }

    #[test]
    fn from_rows() {
        let m = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m[[1, 2]], 6);
    }

    #[test]
    fn transpose() {
        let m = Matrix::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let mt = m.transpose();
        assert_eq!(mt.shape(), &[3, 2]);
        assert_eq!(mt[[0, 0]], 1.0);
        assert_eq!(mt[[2, 0]], 3.0);
        assert_eq!(mt[[0, 1]], 4.0);
    }

    #[test]
    fn matmul() {
        let a = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_rows(&[vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.matmul(&b);
        assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matvec() {
        let m = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let v = Vector::from_vec(vec![5.0, 6.0]);
        let result = m.matvec(&v);
        assert_eq!(result.as_slice(), &[17.0, 39.0]);
    }

    #[test]
    fn row_col_extraction() {
        let m = Matrix::from_rows(&[vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(m.row(0).as_slice(), &[1, 2, 3]);
        assert_eq!(m.row(1).as_slice(), &[4, 5, 6]);
        assert_eq!(m.col(0).as_slice(), &[1, 4]);
        assert_eq!(m.col(2).as_slice(), &[3, 6]);
    }

    #[test]
    fn identity_matmul() {
        let m = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let eye = Matrix::<f64>::identity(2);
        assert_eq!(m.matmul(&eye), m);
        assert_eq!(eye.matmul(&m), m);
    }

    #[test]
    #[should_panic(expected = "inner dimensions mismatch")]
    fn matmul_dimension_mismatch() {
        let a = Matrix::<f64>::zeros([2, 3]);
        let b = Matrix::<f64>::zeros([2, 3]);
        let _ = a.matmul(&b);
    }
}
