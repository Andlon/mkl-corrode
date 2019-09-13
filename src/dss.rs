use std::marker::PhantomData;
use mkl_sys::{MklInt, _MKL_DSS_HANDLE_t, MKL_DSS_DEFAULTS, MKL_DSS_ZERO_BASED_INDEXING,
              MKL_DSS_SYMMETRIC_STRUCTURE, MKL_DSS_SYMMETRIC, MKL_DSS_NON_SYMMETRIC,
              MKL_DSS_AUTO_ORDER,
              dss_create_, dss_delete_, dss_define_structure_, dss_reorder_};
use std::ptr::{null_mut, null};

/// A wrapper around _MKL_DSS_HANDLE_t.
///
/// This is not exported from the library, but instead only used to simplify correct
/// destruction when a handle goes out of scope across the symbolic factorization
/// and numerical factorization.
struct Handle {
    handle: _MKL_DSS_HANDLE_t,
    /// Currently we store the options used to construct the handle,
    /// and use the same options when deleting it. TODO: Is this correct?
    opts: MklInt
}

impl Handle {
    fn create(options: MklInt) -> Self {
        let mut handle = null_mut();

        // TODO: Handle errors
        unsafe {
            let error = dss_create_(&mut handle, &options);
            if error != 0 {
                eprintln!("dss_create error: {}", error);
            }
        }
        Self {
            handle,
            opts: options
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            // TODO: Better handling here, but we cannot really do anything else than panic,
            // can we?
            let delete_opts = MKL_DSS_DEFAULTS as MklInt;
            let error = dss_delete_(&mut self.handle, &delete_opts);
            if error != 0 {
                panic!("dss_delete error: {}", error);
            }
        }
    }
}

pub struct DssSolver<T> {
    marker: PhantomData<T>
}

// TODO: Support complex numbers
pub enum MatrixStructure {
    StructurallySymmetric,
    Symmetric,
    NonSymmetric
}

impl MatrixStructure {
    fn to_mkl_opt(&self) -> MklInt {
        use MatrixStructure::*;
        match self {
            StructurallySymmetric => MKL_DSS_SYMMETRIC_STRUCTURE as MklInt,
            Symmetric => MKL_DSS_SYMMETRIC as MklInt,
            NonSymmetric => MKL_DSS_NON_SYMMETRIC as MklInt
        }
    }
}

pub fn check_csr(row_ptr: &[MklInt], columns: &[MklInt]) {
    assert!(row_ptr.len() > 0, "row_ptr must always have positive length.");

    // TODO: Turn into Result and return Result in `from_csr`

    // TODO: Consider explicitly checking that diagonals are explicitly stored?
    // This is necessary for use in the solver
    // Also check that all values are in bounds?
    // Or does MKL do this anyway? Test...

}

mod private { pub trait Sealed {} }

/// Marker trait for supported scalar types.
///
/// Can not be implemented by dependent crates.
pub unsafe trait SupportedScalar: Copy + private::Sealed {}

// TODO: To support f32 we need to pass appropriate options during handle creation
// Can have the sealed trait provide us with the appropriate option for this!
//impl private::Sealed for f32 {}
impl private::Sealed for f64 {}
//unsafe impl SupportedScalar for f32 {}
unsafe impl SupportedScalar for f64 {}

pub struct SymbolicFactorization<T> {
    handle: Handle,
    marker: PhantomData<T>
}

impl<T> SymbolicFactorization<T>
where
    T: SupportedScalar
{
    pub fn from_csr(row_ptr: &[MklInt], columns: &[MklInt], structure: MatrixStructure) -> Self {
        let create_opts = (MKL_DSS_DEFAULTS + MKL_DSS_ZERO_BASED_INDEXING) as MklInt;
        let mut handle = Handle::create(create_opts);

        let define_opts = structure.to_mkl_opt();

        let nnz = columns.len() as MklInt;

        unsafe {
            // TODO: Handle errors
            let error = dss_define_structure_(&mut handle.handle,
                                  &define_opts,
                                  row_ptr.as_ptr(),
                                  &(row_ptr.len() as MklInt - 1),
                                  &(columns.len() as MklInt),
                                  columns.as_ptr(),
                                  &nnz
            );
            if error != 0 {
                eprintln!("dss_define_structure_ error: {}", error);
            }
        }

        let reorder_opts = MKL_DSS_AUTO_ORDER as MklInt;

        unsafe {
            // TODO: Handle errors
            let error = dss_reorder_(&mut handle.handle, &reorder_opts, null());
            if error != 0 {
                eprintln!("dss_reorder_ error: {}", error);
            }
        }

        Self {
            handle,
            marker: PhantomData
        }
    }
}