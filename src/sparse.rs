use crate::util::is_same_type;
use mkl_sys::{
    matrix_descr, mkl_sparse_d_create_csr, mkl_sparse_destroy, sparse_diag_type_t,
    sparse_fill_mode_t, sparse_matrix_t, sparse_matrix_type_t, MKL_INT,
};
use std::marker::PhantomData;
use std::ptr::null_mut;

use crate::SupportedScalar;
use mkl_sys::sparse_status_t;

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
    UnknownError(sparse_status_t::Type),
}

impl SparseStatusCode {
    pub fn from_raw_code(status: sparse_status_t::Type) -> SparseStatusCode {
        assert_ne!(status, sparse_status_t::SPARSE_STATUS_SUCCESS);
        use sparse_status_t::*;
        use SparseStatusCode::*;

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
    routine: &'static str,
}

impl SparseStatusError {
    // TODO: pub (crate) does not look so nice. Rather reorganize modules?
    pub(crate) fn new_result(
        code: sparse_status_t::Type,
        routine: &'static str,
    ) -> Result<(), Self> {
        if code == sparse_status_t::SPARSE_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Self {
                code: SparseStatusCode::from_raw_code(code),
                routine,
            })
        }
    }

    pub fn code(&self) -> SparseStatusCode {
        self.code
    }

    pub fn routine(&self) -> &str {
        &self.routine
    }
}

pub struct CsrMatrixHandle<'a, T> {
    marker: PhantomData<&'a T>,
    rows: usize,
    cols: usize,
    pub(crate) handle: sparse_matrix_t,
}

impl<'a, T> Drop for CsrMatrixHandle<'a, T> {
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
where
    T: SupportedScalar,
{
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn from_csr_data(
        rows: usize,
        cols: usize,
        row_begin: &'a [MKL_INT],
        row_end: &'a [MKL_INT],
        columns: &'a [MKL_INT],
        values: &'a [T]
    ) -> Result<Self, SparseStatusError> {
        assert_eq!(row_begin.len(), rows, "row_begin length and rows must be equal");
        assert_eq!(row_end.len(), rows, "row_end length and rows must be equal");
        assert_eq!(columns.len(), values.len(), "columns and values must have equal length");

        assert_eq!(row_begin.first().unwrap_or(&0), &0);
        assert!(row_end.first().unwrap_or(&0) >= &0);

        let is_monotonic = |slice: &[_]| (1 .. slice.len())
            .all(|row_idx| slice[row_idx] >= slice[row_idx - 1]);
        assert!(is_monotonic(row_begin));
        assert!(is_monotonic(row_end));

        // Since the numbers are monotonic, it follows that the last element is
        // non-negative, and therefore fits inside a usize
        assert_eq!(row_end.last().copied().unwrap_or(0) as usize, values.len());

        // TODO: Do column indices in each row need to be sorted? I don't think so...
        // MKL docs don't say that in the description of the format, at least.
        assert!(columns.iter().all(|index| index >= &0 && (*index as usize) < cols),
                "column indices must lie in the interval [0, cols)");

        unsafe {
            Self::from_raw_csr_data(rows, cols, row_begin, row_end, columns, values)
        }
    }

    /// TODO: Change this to be more general?
    /// TODO: Build safe abstraction on top
    pub unsafe fn from_raw_csr_data(
        rows: usize,
        cols: usize,
        row_begin: &'a [MKL_INT],
        row_end: &'a [MKL_INT],
        columns: &'a [MKL_INT],
        values: &'a [T],
    ) -> Result<Self, SparseStatusError> {
        // TODO: Handle this more properly
        let rows_mkl = rows as MKL_INT;
        let cols_mkl = cols as MKL_INT;

        let mut handle = null_mut();
        if is_same_type::<T, f64>() {
            // Note: According to
            // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-create-csr
            // MKL does not modify the input arrays UNLESS we call mkl_sparse_order,
            // so it should be safe to borrow the data as long as we don't do that.
            let status = mkl_sparse_d_create_csr(
                &mut handle,
                0,
                rows_mkl,
                cols_mkl,
                row_begin.as_ptr() as *mut _,
                row_end.as_ptr() as *mut _,
                columns.as_ptr() as *mut _,
                values.as_ptr() as *mut _,
            );
            SparseStatusError::new_result(status, "mkl_sparse_d_create_csr")?;
            Ok(Self {
                marker: PhantomData,
                rows,
                cols,
                handle,
            })
        } else {
            // TODO: Implement more types
            panic!("Unsupported type")
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparseMatrixType {
    General,
    Symmetric,
}

impl SparseMatrixType {
    fn to_mkl_value(&self) -> sparse_matrix_type_t::Type {
        match self {
            SparseMatrixType::General => sparse_matrix_type_t::SPARSE_MATRIX_TYPE_GENERAL,
            SparseMatrixType::Symmetric => sparse_matrix_type_t::SPARSE_MATRIX_TYPE_SYMMETRIC,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparseFillMode {
    Lower,
    Upper,
}

impl SparseFillMode {
    fn to_mkl_value(&self) -> sparse_fill_mode_t::Type {
        match self {
            SparseFillMode::Lower => sparse_fill_mode_t::SPARSE_FILL_MODE_LOWER,
            SparseFillMode::Upper => sparse_fill_mode_t::SPARSE_FILL_MODE_UPPER,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparseDiagType {
    NonUnit,
    Unit,
}

impl SparseDiagType {
    fn to_mkl_value(&self) -> sparse_diag_type_t::Type {
        match self {
            SparseDiagType::NonUnit => sparse_diag_type_t::SPARSE_DIAG_NON_UNIT,
            SparseDiagType::Unit => sparse_diag_type_t::SPARSE_DIAG_UNIT,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MatrixDescription {
    matrix_type: SparseMatrixType,
    fill_mode: SparseFillMode,
    diag_type: SparseDiagType,
}

impl Default for MatrixDescription {
    fn default() -> Self {
        Self {
            matrix_type: SparseMatrixType::General,
            fill_mode: SparseFillMode::Upper,
            diag_type: SparseDiagType::NonUnit,
        }
    }
}

impl MatrixDescription {
    pub fn with_type(self, matrix_type: SparseMatrixType) -> Self {
        Self {
            matrix_type,
            ..self
        }
    }

    pub fn with_fill_mode(self, fill_mode: SparseFillMode) -> Self {
        Self { fill_mode, ..self }
    }

    pub fn with_diag_type(self, diag_type: SparseDiagType) -> Self {
        Self { diag_type, ..self }
    }

    pub(crate) fn to_mkl_descr(&self) -> matrix_descr {
        matrix_descr {
            type_: self.matrix_type.to_mkl_value(),
            mode: self.fill_mode.to_mkl_value(),
            diag: self.diag_type.to_mkl_value(),
        }
    }
}
