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

    /// Squared magnitude (avoids the sqrt).
    pub fn magnitude_squared(&self) -> T {
        self.inner_product(self)
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

    /// Cross product (3D or 7D vectors).
    ///
    /// # Panics
    /// Panics if vectors are not length 3 or 7, or if lengths differ.
    pub fn cross(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape[0], other.shape[0],
            "cross product requires equal-length vectors"
        );
        let n = self.shape[0];
        assert!(
            n == 3 || n == 7,
            "cross product is only defined for 3D and 7D vectors, got {n}D"
        );
        match n {
            3 => {
                // Derive from the antisymmetric part of the outer product:
                // A = outer(a, b) - outer(a, b)^T, then cross = [A[1,2], A[2,0], A[0,1]]
                let m = self.outer_product(other);
                let mt = m.transpose();
                let a_mat = &m - &mt;
                Self {
                    shape: [3],
                    data: vec![a_mat[[1, 2]], a_mat[[2, 0]], a_mat[[0, 1]]],
                }
            }
            7 => {
                // 7D cross product uses octonion structure constants — kept as explicit formula.
                let (a, b) = (&self.data, &other.data);
                Self {
                    shape: [7],
                    data: vec![
                        a[1] * b[3] - a[3] * b[1] + a[2] * b[6] - a[6] * b[2] + a[4] * b[5] - a[5] * b[4],
                        a[2] * b[4] - a[4] * b[2] + a[3] * b[0] - a[0] * b[3] + a[5] * b[6] - a[6] * b[5],
                        a[3] * b[5] - a[5] * b[3] + a[4] * b[1] - a[1] * b[4] + a[6] * b[0] - a[0] * b[6],
                        a[4] * b[6] - a[6] * b[4] + a[5] * b[2] - a[2] * b[5] + a[0] * b[1] - a[1] * b[0],
                        a[5] * b[0] - a[0] * b[5] + a[6] * b[3] - a[3] * b[6] + a[1] * b[2] - a[2] * b[1],
                        a[6] * b[1] - a[1] * b[6] + a[0] * b[4] - a[4] * b[0] + a[2] * b[3] - a[3] * b[2],
                        a[0] * b[2] - a[2] * b[0] + a[1] * b[5] - a[5] * b[1] + a[3] * b[4] - a[4] * b[3],
                    ],
                }
            }
            _ => unreachable!(),
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
    fn cross_product_7d() {
        // e1 x e2 should equal e4 in the standard 7D cross product
        let mut e1 = vec![0.0_f64; 7];
        let mut e2 = vec![0.0_f64; 7];
        e1[0] = 1.0;
        e2[1] = 1.0;
        let a = Vector::from_vec(e1);
        let b = Vector::from_vec(e2);
        let c = a.cross(&b);
        // Result should be e4 (index 3)
        assert_eq!(c.as_slice(), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        // Cross product should be anti-commutative: a x b = -(b x a)
        let d = b.cross(&a);
        assert_eq!(d.as_slice(), &[0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "only defined for 3D and 7D")]
    fn cross_product_bad_dimension() {
        let a = Vector::from_vec(vec![1.0, 2.0]);
        let b = Vector::from_vec(vec![3.0, 4.0]);
        let _ = a.cross(&b);
    }

    #[test]
    #[should_panic(expected = "cannot normalize a zero vector")]
    fn normalize_zero() {
        let v = Vector::from_vec(vec![0.0_f64, 0.0, 0.0]);
        let _ = v.normalized();
    }
}
