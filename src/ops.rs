use std::ops::{Add, Mul, Neg, Sub};

use crate::numeric::Numeric;
use crate::tensor::Tensor;

// -- Element-wise binary ops --

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        // owned + owned
        impl<T: Numeric, const R: usize> $trait for Tensor<T, R> {
            type Output = Tensor<T, R>;

            fn $method(self, rhs: Tensor<T, R>) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "shape mismatch in {}", stringify!($trait));
                let data = self.data.iter()
                    .zip(rhs.data.iter())
                    .map(|(&a, &b)| a $op b)
                    .collect();
                Tensor { shape: self.shape, data }
            }
        }

        // &tensor + &tensor
        impl<T: Numeric, const R: usize> $trait for &Tensor<T, R> {
            type Output = Tensor<T, R>;

            fn $method(self, rhs: &Tensor<T, R>) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "shape mismatch in {}", stringify!($trait));
                let data = self.data.iter()
                    .zip(rhs.data.iter())
                    .map(|(&a, &b)| a $op b)
                    .collect();
                Tensor { shape: self.shape, data }
            }
        }

        // owned + &tensor
        impl<T: Numeric, const R: usize> $trait<&Tensor<T, R>> for Tensor<T, R> {
            type Output = Tensor<T, R>;

            fn $method(self, rhs: &Tensor<T, R>) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "shape mismatch in {}", stringify!($trait));
                let data = self.data.iter()
                    .zip(rhs.data.iter())
                    .map(|(&a, &b)| a $op b)
                    .collect();
                Tensor { shape: self.shape, data }
            }
        }

        // &tensor + owned
        impl<T: Numeric, const R: usize> $trait<Tensor<T, R>> for &Tensor<T, R> {
            type Output = Tensor<T, R>;

            fn $method(self, rhs: Tensor<T, R>) -> Self::Output {
                assert_eq!(self.shape, rhs.shape, "shape mismatch in {}", stringify!($trait));
                let data = self.data.iter()
                    .zip(rhs.data.iter())
                    .map(|(&a, &b)| a $op b)
                    .collect();
                Tensor { shape: self.shape, data }
            }
        }
    };
}

impl_binop!(Add, add, +);
impl_binop!(Sub, sub, -);
impl_binop!(Mul, mul, *);

// -- Negation --

impl<T: Numeric, const R: usize> Neg for Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn neg(self) -> Self::Output {
        let data = self.data.iter().map(|&v| -v).collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

impl<T: Numeric, const R: usize> Neg for &Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn neg(self) -> Self::Output {
        let data = self.data.iter().map(|&v| -v).collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

// -- Scalar multiplication: Tensor<T> * T --

impl<T: Numeric, const R: usize> Mul<T> for Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn mul(self, scalar: T) -> Self::Output {
        let data = self.data.iter().map(|&v| v * scalar).collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

impl<T: Numeric, const R: usize> Mul<T> for &Tensor<T, R> {
    type Output = Tensor<T, R>;

    fn mul(self, scalar: T) -> Self::Output {
        let data = self.data.iter().map(|&v| v * scalar).collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn add_sub_mul() {
        let a = Tensor::from_shape_vec([3], vec![1.0, 2.0, 3.0]);
        let b = Tensor::from_shape_vec([3], vec![4.0, 5.0, 6.0]);

        let sum = &a + &b;
        assert_eq!(sum.as_slice(), &[5.0, 7.0, 9.0]);

        let diff = &a - &b;
        assert_eq!(diff.as_slice(), &[-3.0, -3.0, -3.0]);

        let prod = &a * &b;
        assert_eq!(prod.as_slice(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn neg() {
        let a = Tensor::from_shape_vec([2], vec![1, -2]);
        let n = -a;
        assert_eq!(n.as_slice(), &[-1, 2]);
    }

    #[test]
    fn scalar_mul() {
        let a = Tensor::from_shape_vec([3], vec![1.0, 2.0, 3.0]);
        let scaled = a * 2.0;
        assert_eq!(scaled.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_shape_mismatch() {
        let a = Tensor::from_shape_vec([2], vec![1.0, 2.0]);
        let b = Tensor::from_shape_vec([3], vec![1.0, 2.0, 3.0]);
        let _ = a + b;
    }
}
