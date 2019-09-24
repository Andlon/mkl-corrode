use mkl_sys::{MKL_INT, mkl_sparse_d_create_csr, sparse_matrix_t, mkl_sparse_destroy};
use std::marker::PhantomData;
use std::ptr::null_mut;
use crate::util::is_same_type;

use mkl_sys::{sparse_status_t};

mod internal {
    use core::fmt;

    pub trait InternalScalar: 'static + fmt::Debug {}

    impl InternalScalar for f64 {}
}

pub trait SupportedScalar: internal::InternalScalar {}
impl SupportedScalar for f64 {}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparseStatusCode {
    NotInitialized,
    InvalidValue,
    InternalError,

    // TODO: More errors

    /// Special enum variant that corresponds to an error returned by MKL that is not recognized
    /// by the `mkl-corrode` crate.
    ///
    /// This can happen if e.g. a new version of MKL adds new possible return values.
    /// The integer returned is the status code that was not recognized.
    UnknownError(sparse_status_t::Type)
}

impl SparseStatusCode {
    pub fn from_raw_code(status: sparse_status_t::Type) -> SparseStatusCode {
        assert_ne!(status, sparse_status_t::SPARSE_STATUS_SUCCESS);
        use SparseStatusCode::*;
        use sparse_status_t::*;

        if status == SPARSE_STATUS_NOT_INITIALIZED {
            NotInitialized
        } else if status == SPARSE_STATUS_INVALID_VALUE {
            InvalidValue
        } else if status == SPARSE_STATUS_INTERNAL_ERROR {
            InternalError
        } else {
            UnknownError(status)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseStatusError {
    code: SparseStatusCode,
    routine: &'static str
}

impl SparseStatusError {
    fn new_result(code: sparse_status_t::Type, routine: &'static str) -> Result<(), Self> {
        if code == sparse_status_t::SPARSE_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Self { code: SparseStatusCode::from_raw_code(code), routine })
        }
    }

    pub fn code(&self) -> SparseStatusCode {
        self.code
    }

    pub fn routine(&self) -> &str {
        &self.routine
    }
}

pub struct CsrMatrixHandle<'a, T>
{
    marker: PhantomData<&'a T>,
    handle: sparse_matrix_t
}

impl<'a, T> Drop for CsrMatrixHandle<'a, T>
{
    fn drop(&mut self) {
        unsafe {
            // TODO: Does MKL actually take ownership of the arrays in _create_csr?
            // In other words, will this try to deallocate the memory of the matrices passed in
            // as slices? If so, that would be disastrous. The Intel MKL docs are as usual
            // not clear on this
            let status = mkl_sparse_destroy(self.handle);
            if SparseStatusError::new_result(status, "mkl_sparse_destroy").is_err() {
                // TODO: Should we panic here? Or just print to eprintln!?
                // I'd venture that if this fails, then there's something seriously wrong
                // somewhere...
                panic!("Error during sparse matrix destruction.")
            };
        }
    }
}

impl<'a, T> CsrMatrixHandle<'a, T>
where T: SupportedScalar
{
    /// TODO: Change this to be more general?
    /// TODO: Build safe abstraction on top
    pub unsafe fn from_raw_csr_data(rows: usize,
                                    cols: usize,
                                    row_begin: &'a [MKL_INT],
                                    row_end: &'a [MKL_INT],
                                    columns: &'a [MKL_INT],
                                    values: &'a [T])
        -> Result<Self, SparseStatusError>
    {
        // TODO: Handle this more properly
        let rows = rows as MKL_INT;
        let cols = cols as MKL_INT;

        let mut handle = null_mut();
        if is_same_type::<T, f64>() {
            // Note: According to
            // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
            // MKL does not modify the input arrays UNLESS we call mkl_sparse_order,
            // so it should be safe to borrow the data as long as we don't do that.
            let status = mkl_sparse_d_create_csr(&mut handle, 0, rows, cols,
                                                 row_begin.as_ptr() as *mut _,
                                                 row_end.as_ptr() as *mut _,
                                                 columns.as_ptr() as *mut _,
                                                 values.as_ptr() as *mut _);
            SparseStatusError::new_result(status, "mkl_sparse_d_create_csr")?;
            Ok(Self {
                marker: PhantomData,
                handle
            })
        } else {
            // TODO: Implement more types
            panic!("Unsupported type")
        }
    }

}
