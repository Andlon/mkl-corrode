use mkl_sys::{
    MKL_INT, _MKL_DSS_HANDLE_t, dss_create_, dss_define_structure_, dss_delete_, dss_factor_real_,
    dss_reorder_, dss_solve_real_, MKL_DSS_AUTO_ORDER, MKL_DSS_BACKWARD_SOLVE, MKL_DSS_DEFAULTS,
    MKL_DSS_DIAGONAL_SOLVE, MKL_DSS_FORWARD_SOLVE, MKL_DSS_INDEFINITE, MKL_DSS_NON_SYMMETRIC,
    MKL_DSS_POSITIVE_DEFINITE, MKL_DSS_SYMMETRIC, MKL_DSS_SYMMETRIC_STRUCTURE,
    MKL_DSS_ZERO_BASED_INDEXING,
};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::{null, null_mut};

// MKL constants
use mkl_sys::{
    MKL_DSS_COL_ERR, MKL_DSS_DIAG_ERR, MKL_DSS_FAILURE, MKL_DSS_I32BIT_ERR, MKL_DSS_INVALID_OPTION,
    MKL_DSS_MSG_LVL_ERR, MKL_DSS_NOT_SQUARE, MKL_DSS_OOC_MEM_ERR, MKL_DSS_OOC_OC_ERR,
    MKL_DSS_OOC_RW_ERR, MKL_DSS_OPTION_CONFLICT, MKL_DSS_OUT_OF_MEMORY, MKL_DSS_REORDER1_ERR,
    MKL_DSS_REORDER_ERR, MKL_DSS_ROW_ERR, MKL_DSS_STATE_ERR, MKL_DSS_STATISTICS_INVALID_MATRIX,
    MKL_DSS_STATISTICS_INVALID_STATE, MKL_DSS_STATISTICS_INVALID_STRING, MKL_DSS_STRUCTURE_ERR,
    MKL_DSS_SUCCESS, MKL_DSS_TERM_LVL_ERR, MKL_DSS_TOO_FEW_VALUES, MKL_DSS_TOO_MANY_VALUES,
    MKL_DSS_VALUES_ERR,
};
use std::borrow::Cow;
use std::convert::TryFrom;
use std::any::{TypeId};
use std::mem::transmute;
use std::fmt::{Display, Debug};
use core::fmt;

/// Calls the given DSS function, noting its error code and upon a non-success result,
/// returns an appropriate error.
macro_rules! dss_call {
    ($routine:ident ($($arg: tt)*)) => {
        {
            let code = $routine($($arg)*);
            if code != MKL_DSS_SUCCESS {
                return Err(Error::new(ErrorCode::from_return_code(code), stringify!($routine)));
            }
        }
    }
}

// TODO: We only care about square matrices
#[derive(Debug, PartialEq, Eq)]
pub struct SparseMatrix<'a, T>
where
    T: Clone
{
    row_offsets: Cow<'a, [MKL_INT]>,
    columns: Cow<'a, [MKL_INT]>,
    values: Cow<'a, [T]>,
    structure: MatrixStructure
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SparseMatrixDataError {
    NonMonotoneColumns,
    MissingExplicitDiagonal,
    UnexpectedLowerTriangularPart,
    NonMonotoneRowOffsets,
    EmptyRowOffsets,
    InvalidRowOffset,
    InvalidColumnIndex,
    InsufficientIndexSize
}

impl SparseMatrixDataError {
    fn is_recoverable(&self) -> bool {
        use SparseMatrixDataError::*;
        match self {
            NonMonotoneColumns => false,
            MissingExplicitDiagonal => true,
            UnexpectedLowerTriangularPart => true,
            NonMonotoneRowOffsets => false,
            EmptyRowOffsets => false,
            InvalidRowOffset => false,
            InvalidColumnIndex => false,
            InsufficientIndexSize => false
        }
    }
}

impl Display for SparseMatrixDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Error in sparse matrix data: {:?}", self)
    }
}

impl std::error::Error for SparseMatrixDataError {

}

fn is_same_type<T, U>() -> bool
where
    T: 'static,
    U: 'static
{
    TypeId::of::<T>() == TypeId::of::<U>()
}

// TODO: Move to utils file or something?
fn transmute_identical_slice<T, U>(slice: &[T]) -> Option<&[U]>
    where T: 'static, U: 'static {
    if is_same_type::<T, U>() {
        Some(unsafe { transmute(slice) })
    } else {
        None
    }
}

trait CsrProcessor<T> {
    /// Called when processing of the current row has finished.
    fn row_processed(&mut self) {}
    fn visit_column(&mut self, i: MKL_INT, j: MKL_INT, v: &T) -> Result<(), SparseMatrixDataError>;
    fn visit_missing_diagonal_entry(&mut self, i: MKL_INT) -> Result<(), SparseMatrixDataError>;
}

fn process_csr<'a, T, I>(row_offsets: &'a [I],
                         columns: &'a [I],
                         values: &'a [T],
                         structure: MatrixStructure,
                         processor: &mut impl CsrProcessor<T>)
                         -> Result<(), SparseMatrixDataError>
    where
        T: SupportedScalar,
        usize: TryFrom<I>,
        MKL_INT: TryFrom<I>,
        I: Copy
{
    let needs_explicit_diagonal = match structure {
        MatrixStructure::Symmetric | MatrixStructure::StructurallySymmetric => false,
        MatrixStructure::NonSymmetric => true
    };

    // Helper conversion functions.
    let offset_as_usize = |offset| usize::try_from(offset)
        .map_err(|_| SparseMatrixDataError::InvalidRowOffset);
    let index_as_mkl_int = |idx| MKL_INT::try_from(idx)
        .map_err(|_| SparseMatrixDataError::InvalidColumnIndex);
    let usize_as_mkl_int = |idx| <MKL_INT as TryFrom<usize>>::try_from(idx)
        .map_err(|_| SparseMatrixDataError::InsufficientIndexSize);

    let num_rows = row_offsets.len() - 1;
    let num_cols = usize_as_mkl_int(num_rows)?;
    let nnz = values.len();
    // TODO: Assertion or error?
    assert_eq!(nnz, columns.len());

    if row_offsets.is_empty() {
        return Err(SparseMatrixDataError::EmptyRowOffsets);
    }

    if nnz != offset_as_usize(*row_offsets.last().unwrap())? {
        return Err(SparseMatrixDataError::InvalidRowOffset);
    }

    for i in 0..num_rows {
        let current_offset = row_offsets[i];
        let row_begin = offset_as_usize(current_offset)?;
        let row_end = offset_as_usize(row_offsets[i + 1])?;
        let i = usize_as_mkl_int(i)?;

        if row_end < row_begin {
            return Err(SparseMatrixDataError::NonMonotoneRowOffsets);
        }

        // - check that each column is in bounds, if not abort
        // - check that column indices are monotone increasing, if not abort
        // - If (structurally) symmetric: check that the diagonal element exists, if not insert it
        // - If (structurally) symmetric: ignore lower triangular elements

        let columns_for_row = &columns[row_begin..row_end];
        let values_for_row = &values[row_begin..row_end];

        // TODO: Rename to "have_processed"
        let mut have_placed_diagonal = false;
        let mut prev_column = None;
        for (j, v_j) in columns_for_row.iter().zip(values_for_row) {
            let j = index_as_mkl_int(*j)?;

            if j < 0 || j >= num_cols {
                return Err(SparseMatrixDataError::InvalidColumnIndex);
            }

            if let Some(j_prev) = prev_column {
                if j <= j_prev {
                    return Err(SparseMatrixDataError::NonMonotoneColumns);
                }
            }

            if needs_explicit_diagonal {
                if i == j {
                    have_placed_diagonal = true;
                    // TODO: Can remove the i < j comparison here!
                } else if i < j && !have_placed_diagonal {
                    processor.visit_missing_diagonal_entry(i)?;
                    have_placed_diagonal = true;
                }
            }

            processor.visit_column(i, j, v_j)?;
            prev_column = Some(j);
        }
        processor.row_processed();
    }
    Ok(())
}

fn rebuild_csr<'a, T, I>(row_offsets: &'a [I],
                     columns: &'a [I],
                     values: &'a [T],
                     structure: MatrixStructure)
    -> Result<SparseMatrix<'a, T>, SparseMatrixDataError>
where
    T: SupportedScalar,
    usize: TryFrom<I>,
    MKL_INT: TryFrom<I>,
    I: Copy
{
    let keep_lower_tri = match structure {
        MatrixStructure::Symmetric | MatrixStructure::StructurallySymmetric => false,
        MatrixStructure::NonSymmetric => true
    };

    struct CsrRebuilder<X> {
        new_row_offsets: Vec<MKL_INT>,
        new_columns: Vec<MKL_INT>,
        new_values: Vec<X>,
        current_offset: MKL_INT,
        num_cols_in_current_row: MKL_INT,
        keep_lower_tri: bool
    }

    impl<X> CsrRebuilder<X> {
        fn push_val(&mut self, j: MKL_INT, v_j: X) {
            self.new_columns.push(j);
            self.new_values.push(v_j);
            self.num_cols_in_current_row += 1;
        }
    }

    impl<X: SupportedScalar> CsrProcessor<X> for CsrRebuilder<X> {
        fn row_processed(&mut self) {
            let new_offset = self.current_offset + self.num_cols_in_current_row;
            self.current_offset = new_offset;
            self.num_cols_in_current_row = 0;
            self.new_row_offsets.push(new_offset);
        }

        fn visit_column(&mut self, i: i32, j: i32, v_j: &X) -> Result<(), SparseMatrixDataError> {
            let should_push = j >= i || (j < i && self.keep_lower_tri);
            if should_push {
                self.push_val(j, *v_j);
            }
            Ok(())
        }

        fn visit_missing_diagonal_entry(&mut self, i: i32) -> Result<(), SparseMatrixDataError> {
            self.push_val(i, X::zero_element());
            Ok(())
        }
    }

    let mut rebuilder = CsrRebuilder {
        new_row_offsets: vec![0],
        new_columns: Vec::new(),
        new_values: Vec::new(),
        current_offset: 0,
        num_cols_in_current_row: 0,
        keep_lower_tri
    };

    process_csr(row_offsets, columns, values, structure, &mut rebuilder)?;

    let matrix = SparseMatrix {
        row_offsets: Cow::Owned(rebuilder.new_row_offsets),
        columns: Cow::Owned(rebuilder.new_columns),
        values: Cow::Owned(rebuilder.new_values),
        structure
    };
    Ok(matrix)
}

impl<'a, T> SparseMatrix<'a, T>
where
    T: SupportedScalar
{
    pub fn row_offsets(&self) -> &[MKL_INT] {
        &self.row_offsets
    }

    pub fn columns(&self) -> &[MKL_INT] {
        &self.columns
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn structure(&self) -> MatrixStructure {
        self.structure
    }

    pub fn try_from_csr(row_offsets: &'a [MKL_INT],
                        columns: &'a [MKL_INT],
                        values: &'a [T],
                        structure: MatrixStructure)
        -> Result<Self, SparseMatrixDataError>
    {
        let allow_lower_tri = match structure {
            MatrixStructure::Symmetric | MatrixStructure::StructurallySymmetric => false,
            MatrixStructure::NonSymmetric => true
        };

        struct CsrCheck {
            allow_lower_tri: bool
        }

        impl<X: SupportedScalar> CsrProcessor<X> for CsrCheck {
            fn visit_column(&mut self, i: i32, j: i32, _: &X) -> Result<(), SparseMatrixDataError> {
                if !self.allow_lower_tri && j < i {
                    Err(SparseMatrixDataError::UnexpectedLowerTriangularPart)
                } else {
                    Ok(())
                }
            }

            fn visit_missing_diagonal_entry(&mut self, _: i32) -> Result<(), SparseMatrixDataError> {
                Err(SparseMatrixDataError::MissingExplicitDiagonal)
            }
        }

        let mut checker = CsrCheck { allow_lower_tri };
        process_csr(row_offsets, columns, values, structure, &mut checker)?;

        let matrix = SparseMatrix {
            row_offsets: Cow::Borrowed(row_offsets),
            columns: Cow::Borrowed(columns),
            values: Cow::Borrowed(values),
            structure
        };
        Ok(matrix)
    }

    pub fn try_convert_from_csr<I>(row_offsets: &'a [I],
                                   columns: &'a [I],
                                   values: &'a [T],
                                   structure: MatrixStructure)
        -> Result<Self, SparseMatrixDataError>
    where
        I: 'static + Copy,
        MKL_INT: TryFrom<I>,
        usize: TryFrom<I>
    {
        // If the data already has the right integer type, then try to pass it in to MKL directly.
        // If it fails, it might be that we can recover by rebuilding the matrix data.
        if is_same_type::<I, MKL_INT>() {
            let row_offsets_mkl_int = transmute_identical_slice(row_offsets).unwrap();
            let columns_mkl_int = transmute_identical_slice(columns).unwrap();
            let result = Self::try_from_csr(row_offsets_mkl_int, columns_mkl_int, values, structure);
            match result {
                Ok(matrix) => return Ok(matrix),
                Err(error) => if !error.is_recoverable() {
                    return Err(error);
                }
            }
        };

        rebuild_csr(row_offsets, columns, values, structure)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Error {
    code: ErrorCode,
    routine: &'static str
}

impl Error {
    pub fn return_code(&self) -> ErrorCode {
        self.code
    }

    pub fn routine(&self) -> &str {
        self.routine
    }

    fn new(code: ErrorCode, routine: &'static str) -> Self {
        Self { code, routine }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Error in routine {}. Return code: {:?}", self.routine(), self.return_code())
    }
}

impl std::error::Error for Error {

}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    InvalidOption,
    OutOfMemory,
    MsgLvlErr,
    TermLvlErr,
    StateErr,
    RowErr,
    ColErr,
    StructureErr,
    NotSquare,
    ValuesErr,
    TooFewValues,
    TooManyValues,
    ReorderErr,
    Reorder1Err,
    I32BitErr,
    Failure,
    OptionConflict,
    OocMemErr,
    OocOcErr,
    OocRwErr,
    DiagErr,
    StatisticsInvalidMatrix,
    StatisticsInvalidState,
    StatisticsInvalidString,

    /// Special error that does not exist in Intel MKL.
    ///
    /// This error is used when we encounter an unknown return code. This could for example
    /// happen if a new version of Intel MKL adds more return codes and this crate has not
    /// been updated to take that into account.
    UnknownError,
}

impl ErrorCode {
    /// Construct a `DssError` from an MKL return code.
    ///
    /// This should cover every return code possible, but see notes made
    /// in the docs for `UnknownError`.
    fn from_return_code(code: MKL_INT) -> Self {
        assert_ne!(code, MKL_DSS_SUCCESS);

        if code == MKL_DSS_INVALID_OPTION {
            Self::InvalidOption
        } else if code == MKL_DSS_OUT_OF_MEMORY {
            Self::OutOfMemory
        } else if code == MKL_DSS_MSG_LVL_ERR {
            Self::MsgLvlErr
        } else if code == MKL_DSS_TERM_LVL_ERR {
            Self::TermLvlErr
        } else if code == MKL_DSS_STATE_ERR {
            Self::StateErr
        } else if code == MKL_DSS_ROW_ERR {
            Self::RowErr
        } else if code == MKL_DSS_COL_ERR {
            Self::ColErr
        } else if code == MKL_DSS_STRUCTURE_ERR {
            Self::StructureErr
        } else if code == MKL_DSS_NOT_SQUARE {
            Self::NotSquare
        } else if code == MKL_DSS_VALUES_ERR {
            Self::ValuesErr
        } else if code == MKL_DSS_TOO_FEW_VALUES {
            Self::TooFewValues
        } else if code == MKL_DSS_TOO_MANY_VALUES {
            Self::TooManyValues
        } else if code == MKL_DSS_REORDER_ERR {
            Self::ReorderErr
        } else if code == MKL_DSS_REORDER1_ERR {
            Self::Reorder1Err
        } else if code == MKL_DSS_I32BIT_ERR {
            Self::I32BitErr
        } else if code == MKL_DSS_FAILURE {
            Self::Failure
        } else if code == MKL_DSS_OPTION_CONFLICT {
            Self::OptionConflict
        } else if code == MKL_DSS_OOC_MEM_ERR {
            Self::OocMemErr
        } else if code == MKL_DSS_OOC_OC_ERR {
            Self::OocOcErr
        } else if code == MKL_DSS_OOC_RW_ERR {
            Self::OocRwErr
        } else if code == MKL_DSS_DIAG_ERR {
            Self::DiagErr
        } else if code == MKL_DSS_STATISTICS_INVALID_MATRIX {
            Self::StatisticsInvalidMatrix
        } else if code == MKL_DSS_STATISTICS_INVALID_STATE {
            Self::StatisticsInvalidState
        } else if code == MKL_DSS_STATISTICS_INVALID_STRING {
            Self::StatisticsInvalidString
        } else {
            Self::UnknownError
        }
    }
}

/// A wrapper around _MKL_DSS_HANDLE_t.
///
/// This is not exported from the library, but instead only used to simplify correct
/// destruction when a handle goes out of scope across the symbolic factorization
/// and numerical factorization.
struct Handle {
    handle: _MKL_DSS_HANDLE_t,
}

impl Handle {
    fn create(options: MKL_INT) -> Result<Self, Error> {
        let mut handle = null_mut();
        unsafe { dss_call! { dss_create_(&mut handle, &options) }}
        Ok(Self { handle })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            // TODO: Better handling here, but we cannot really do anything else than panic,
            // can we?
            let delete_opts = MKL_DSS_DEFAULTS;
            let error = dss_delete_(&mut self.handle, &delete_opts);
            if error != 0 {
                panic!("dss_delete error: {}", error);
            }
        }
    }
}

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Definiteness {
    PositiveDefinite,
    Indefinite,
}

impl Definiteness {
    fn to_mkl_opt(&self) -> MKL_INT {
        use Definiteness::*;
        match self {
            PositiveDefinite => MKL_DSS_POSITIVE_DEFINITE,
            Indefinite => MKL_DSS_INDEFINITE,
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

pub struct Solver<T> {
    handle: Handle,
    marker: PhantomData<T>,
    num_rows: usize,
    nnz: usize,
}

impl<T> Solver<T>
where
    T: SupportedScalar,
{
    pub fn try_factor(matrix: &SparseMatrix<T>,
                      definiteness: Definiteness) -> Result<Self, Error> {
        let row_ptr = matrix.row_offsets();
        let columns = matrix.columns();
        let values = matrix.values();
        let structure = matrix.structure();
        let nnz = values.len();

        // TODO: Part of error?
        assert_eq!(values.len(), nnz);

        // TODO: Result instead of panic?
        assert!(row_ptr.len() > 0);
        let num_rows = row_ptr.len() - 1;
        let num_cols = num_rows;

        // TODO: Enable tweaking messages!
        let create_opts = MKL_DSS_DEFAULTS + MKL_DSS_ZERO_BASED_INDEXING;
        let mut handle = Handle::create(create_opts)?;

        let define_opts = structure.to_mkl_opt();
        unsafe { dss_call!{ dss_define_structure_(
                &mut handle.handle,
                &define_opts,
                row_ptr.as_ptr(),
                // TODO: What if num_rows, nnz or num_cols > max(MKL_INT)?
                &(num_rows as MKL_INT),
                &(num_cols as MKL_INT),
                columns.as_ptr(),
                &(nnz as MKL_INT),
        ) }}

        let reorder_opts = MKL_DSS_AUTO_ORDER;
        unsafe { dss_call!{ dss_reorder_(&mut handle.handle, &reorder_opts, null()) }};

        let mut factorization = Solver {
            handle,
            num_rows,
            nnz,
            marker: PhantomData
        };
        factorization.refactor(values, definiteness)?;
        Ok(factorization)
    }

    pub fn refactor(&mut self, values: &[T], definiteness: Definiteness) -> Result<(), Error> {
        // TODO: Part of error?
        assert_eq!(values.len(), self.nnz);

        let opts = definiteness.to_mkl_opt();
        unsafe { dss_call!{ dss_factor_real_(
            &mut self.handle.handle,
            &opts,
            values.as_ptr() as *const c_void,
        ) }};
        Ok(())
    }

    // TODO: Would it be safe to only take &self and still hand in a mutable pointer
    // to the handle? We technically don't have any idea what is happening inside
    // MKL, but on the other hand the factorization cannot be accessed from multiple threads,
    // and I think as far as I can tell that the state of the factorization does not change?
    // Unless an error somehow invalidates the handle? Not clear...
    // Note: same for diagonal/backward
    pub fn forward_substitute_into(&mut self, solution: &mut [T], rhs: &[T]) -> Result<(), Error> {
        let num_rhs = rhs.len() / self.num_rows;

        // TODO: Make part of error?
        assert_eq!(
            rhs.len() % self.num_rows,
            0,
            "Number of entries in RHS must be divisible by system size."
        );
        assert_eq!(solution.len(), rhs.len());

        unsafe { dss_call! {
            dss_solve_real_(
                &mut self.handle.handle,
                &(MKL_DSS_FORWARD_SOLVE),
                rhs.as_ptr() as *const c_void,
                // TODO: What if num_rhs > max(MKL_INT)? Absurd situation, but it could maybe
                // lead to undefined behavior, so we need to handle it
                &(num_rhs as MKL_INT),
                solution.as_mut_ptr() as *mut c_void,
            )
        }};
        Ok(())
    }

    pub fn diagonal_substitute_into(&mut self, solution: &mut [T], rhs: &[T]) -> Result<(), Error> {
        let num_rhs = rhs.len() / self.num_rows;

        // TODO: Make part of error?
        assert_eq!(
            rhs.len() % self.num_rows,
            0,
            "Number of entries in RHS must be divisible by system size."
        );
        assert_eq!(solution.len(), rhs.len());

        unsafe { dss_call! {
            dss_solve_real_(
                &mut self.handle.handle,
                &(MKL_DSS_DIAGONAL_SOLVE),
                rhs.as_ptr() as *const c_void,
                // TODO: See other comment about this coercion cast
                &(num_rhs as MKL_INT),
                solution.as_mut_ptr() as *mut c_void,
            )
        } };
        Ok(())
    }

    pub fn backward_substitute_into(&mut self, solution: &mut [T], rhs: &[T]) -> Result<(), Error> {
        let num_rhs = rhs.len() / self.num_rows;

        // TODO: Make part of error?
        assert_eq!(
            rhs.len() % self.num_rows,
            0,
            "Number of entries in RHS must be divisible by system size."
        );
        assert_eq!(solution.len(), rhs.len());

        unsafe { dss_call! {
            dss_solve_real_(
                &mut self.handle.handle,
                &(MKL_DSS_BACKWARD_SOLVE),
                rhs.as_ptr() as *const c_void,
                // TODO: See other comment about num_rhs and `as` cast
                &(num_rhs as MKL_INT),
                solution.as_mut_ptr() as *mut c_void,
            )
        }};
        Ok(())
    }

    /// Convenience function for calling the different substitution phases.
    ///
    /// `buffer` must have same size as `solution`.
    pub fn solve_into(&mut self, solution: &mut [T], buffer: &mut [T], rhs: &[T]) -> Result<(), Error> {
        let y = solution;
        self.forward_substitute_into(y, rhs)?;

        let z = buffer;
        self.diagonal_substitute_into(z, &y)?;

        let x = y;
        self.backward_substitute_into(x, &z)?;

        Ok(())
    }

    /// Convenience function that internally allocates buffer storage and output storage.
    pub fn solve(&mut self, rhs: &[T]) -> Result<Vec<T>, Error>
    {
        let mut solution = vec![T::zero_element(); rhs.len()];
        let mut buffer = vec![T::zero_element(); rhs.len()];
        self.solve_into(&mut solution, &mut buffer, rhs)?;
        Ok(solution)
    }
}
