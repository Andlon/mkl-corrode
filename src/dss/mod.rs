use mkl_sys::{MKL_INT, MKL_DSS_SYMMETRIC_STRUCTURE, MKL_DSS_SYMMETRIC, MKL_DSS_NON_SYMMETRIC};

mod solver;
mod sparse_matrix;
pub use solver::*;
pub use sparse_matrix::*;

// TODO: Support complex numbers
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixStructure {
    StructurallySymmetric,
    Symmetric,
    NonSymmetric,
}

impl MatrixStructure {
    fn to_mkl_opt(&self) -> MKL_INT {
        use MatrixStructure::*;
        match self {
            StructurallySymmetric => MKL_DSS_SYMMETRIC_STRUCTURE,
            Symmetric => MKL_DSS_SYMMETRIC,
            NonSymmetric => MKL_DSS_NON_SYMMETRIC,
        }
    }
}

mod internal {
    pub trait InternalScalar {
        fn zero_element() -> Self;
    }
}

/// Marker trait for supported scalar types.
///
/// Can not be implemented by dependent crates.
pub unsafe trait SupportedScalar: Copy + internal::InternalScalar {

}

// TODO: To support f32 we need to pass appropriate options during handle creation
// Can have the sealed trait provide us with the appropriate option for this!
//impl private::Sealed for f32 {}
impl internal::InternalScalar for f64 {
    fn zero_element() -> Self {
        0.0
    }
}
//unsafe impl SupportedScalar for f32 {}
unsafe impl SupportedScalar for f64 {}
