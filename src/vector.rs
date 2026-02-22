use crate::numeric::Numeric;
use crate::tensor::Tensor;

/// Methods specific to rank-1 tensors (vectors).
impl<T: Numeric> Tensor<T, 1> {
    /// Create a vector from a `Vec<T>`.
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            shape: [len],
            data,
        }
    }

    /// Dot product of two vectors.
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(
            self.shape, other.shape,
            "dot product requires equal-length vectors"
        );
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    /// Squared magnitude (avoids the sqrt).
    pub fn magnitude_squared(&self) -> T {
        self.data.iter().map(|&v| v * v).sum()
    }

    /// Euclidean magnitude (L2 norm).
    pub fn magnitude(&self) -> T {
        self.magnitude_squared().sqrt()
    }

    /// Alias for `magnitude`.
    pub fn norm(&self) -> T {
        self.magnitude()
    }

    /// Returns a unit vector in the same direction.
    ///
    /// # Panics
    /// Panics if the magnitude is zero.
    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        assert!(mag != T::zero(), "cannot normalize a zero vector");
        let data = self.data.iter().map(|&v| v / mag).collect();
        Self {
            shape: self.shape,
            data,
        }
    }

    /// Cross product (3D vectors only).
    ///
    /// # Panics
    /// Panics if either vector is not length 3.
    pub fn cross(&self, other: &Self) -> Self {
        assert_eq!(self.shape[0], 3, "cross product requires 3D vectors");
        assert_eq!(other.shape[0], 3, "cross product requires 3D vectors");
        let (a, b) = (&self.data, &other.data);
        Self {
            shape: [3],
            data: vec![
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Vector;

    #[test]
    fn dot_product() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn magnitude() {
        let v = Vector::from_vec(vec![3.0_f64, 4.0]);
        assert!((v.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn normalized() {
        let v = Vector::from_vec(vec![3.0_f64, 4.0]);
        let n = v.normalized();
        assert!((n.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cross_product() {
        let x = Vector::from_vec(vec![1.0, 0.0, 0.0]);
        let y = Vector::from_vec(vec![0.0, 1.0, 0.0]);
        let z = x.cross(&y);
        assert_eq!(z.as_slice(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "cannot normalize a zero vector")]
    fn normalize_zero() {
        let v = Vector::from_vec(vec![0.0_f64, 0.0, 0.0]);
        let _ = v.normalized();
    }
}
