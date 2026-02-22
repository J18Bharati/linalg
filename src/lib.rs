pub mod numeric;
pub mod tensor;
pub mod ops;
pub mod vector;
pub mod matrix;

pub use numeric::Numeric;
pub use tensor::Tensor;

/// A rank-0 tensor (scalar wrapper).
pub type Scalar<T> = Tensor<T, 0>;

/// A rank-1 tensor (vector).
pub type Vector<T> = Tensor<T, 1>;

/// A rank-2 tensor (matrix).
pub type Matrix<T> = Tensor<T, 2>;
