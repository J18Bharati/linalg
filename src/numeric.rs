use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Numeric:
    Copy
    + Default
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sum
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
}

macro_rules! impl_numeric_float {
    ($($t:ty),*) => {
        $(
            impl Numeric for $t {
                #[inline]
                fn zero() -> Self { 0.0 }
                #[inline]
                fn one() -> Self { 1.0 }
                #[inline]
                fn sqrt(self) -> Self { <$t>::sqrt(self) }
            }
        )*
    };
}

macro_rules! impl_numeric_int {
    ($($t:ty),*) => {
        $(
            impl Numeric for $t {
                #[inline]
                fn zero() -> Self { 0 }
                #[inline]
                fn one() -> Self { 1 }
                #[inline]
                fn sqrt(self) -> Self { (self as f64).sqrt() as Self }
            }
        )*
    };
}

impl_numeric_float!(f32, f64);
impl_numeric_int!(i8, i16, i32, i64, i128, isize);
