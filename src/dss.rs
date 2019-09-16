use std::marker::PhantomData;
use mkl_sys::{MklInt, _MKL_DSS_HANDLE_t, MKL_DSS_DEFAULTS, MKL_DSS_ZERO_BASED_INDEXING,
              MKL_DSS_SYMMETRIC_STRUCTURE, MKL_DSS_SYMMETRIC, MKL_DSS_NON_SYMMETRIC,
              MKL_DSS_AUTO_ORDER, MKL_DSS_POSITIVE_DEFINITE, MKL_DSS_INDEFINITE,
              MKL_DSS_FORWARD_SOLVE, MKL_DSS_DIAGONAL_SOLVE, MKL_DSS_BACKWARD_SOLVE,
              dss_create_, dss_delete_, dss_define_structure_, dss_reorder_, dss_factor_real_,
              dss_solve_real_};
use std::ptr::{null_mut, null};
use std::ffi::c_void;

// MKL constants
use mkl_sys::{
    MKL_DSS_COL_ERR,
    MKL_DSS_DIAG_ERR,
    MKL_DSS_FAILURE,
    MKL_DSS_I32BIT_ERR,
    MKL_DSS_INVALID_OPTION,
    MKL_DSS_MSG_LVL_ERR,
    MKL_DSS_NOT_SQUARE,
    MKL_DSS_OOC_MEM_ERR,
    MKL_DSS_OOC_OC_ERR,
    MKL_DSS_OOC_RW_ERR,
    MKL_DSS_OPTION_CONFLICT,
    MKL_DSS_OUT_OF_MEMORY,
    MKL_DSS_REORDER1_ERR,
    MKL_DSS_REORDER_ERR,
    MKL_DSS_ROW_ERR,
    MKL_DSS_STATE_ERR,
    MKL_DSS_STATISTICS_INVALID_MATRIX,
    MKL_DSS_STATISTICS_INVALID_STATE,
    MKL_DSS_STATISTICS_INVALID_STRING,
    MKL_DSS_STRUCTURE_ERR,
    MKL_DSS_SUCCESS,
    MKL_DSS_TERM_LVL_ERR,
    MKL_DSS_TOO_FEW_VALUES,
    MKL_DSS_TOO_MANY_VALUES,
    MKL_DSS_VALUES_ERR,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DssError {
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
    UnknownError
}

impl DssError {
    /// Construct a `DssError` from an MKL return code.
    ///
    /// This should cover every return code possible, but see notes made
    /// in the docs for `UnknownError`.
    fn from_return_code(code: MklInt) -> Self {
        if code == MKL_DSS_INVALID_OPTION { Self::InvalidOption }
        else if code == MKL_DSS_OUT_OF_MEMORY { Self::OutOfMemory }
        else if code == MKL_DSS_MSG_LVL_ERR { Self::MsgLvlErr }
        else if code == MKL_DSS_TERM_LVL_ERR { Self::TermLvlErr }
        else if code == MKL_DSS_STATE_ERR { Self::StateErr }
        else if code == MKL_DSS_ROW_ERR { Self::RowErr }
        else if code == MKL_DSS_COL_ERR { Self::ColErr }
        else if code == MKL_DSS_STRUCTURE_ERR { Self::StructureErr }
        else if code == MKL_DSS_NOT_SQUARE { Self::NotSquare }
        else if code == MKL_DSS_VALUES_ERR { Self::ValuesErr }
        else if code == MKL_DSS_TOO_FEW_VALUES { Self::TooFewValues }
        else if code == MKL_DSS_TOO_MANY_VALUES { Self::TooManyValues }
        else if code == MKL_DSS_REORDER_ERR { Self::ReorderErr }
        else if code == MKL_DSS_REORDER1_ERR { Self::Reorder1Err }
        else if code == MKL_DSS_I32BIT_ERR { Self::I32BitErr }
        else if code == MKL_DSS_FAILURE { Self::Failure }
        else if code == MKL_DSS_OPTION_CONFLICT { Self::OptionConflict }
        else if code == MKL_DSS_OOC_MEM_ERR { Self::OocMemErr }
        else if code == MKL_DSS_OOC_OC_ERR { Self::OocOcErr }
        else if code == MKL_DSS_OOC_RW_ERR { Self::OocRwErr }
        else if code == MKL_DSS_DIAG_ERR { Self::DiagErr }
        else if code == MKL_DSS_STATISTICS_INVALID_MATRIX { Self::StatisticsInvalidMatrix }
        else if code == MKL_DSS_STATISTICS_INVALID_STATE { Self::StatisticsInvalidState }
        else if code == MKL_DSS_STATISTICS_INVALID_STRING { Self::StatisticsInvalidString }
        else { Self::UnknownError }
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
    fn create(options: MklInt) -> Result<Self, DssError> {
        let mut handle = null_mut();

        // TODO: Handle errors
        let error = unsafe { dss_create_(&mut handle, &options) };
        if error == MKL_DSS_SUCCESS as MklInt {
            Ok(Self { handle })
        } else {
            Err(DssError::from_return_code(error))
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

// TODO: Support complex numbers
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixDefiniteness {
    PositiveDefinite,
    Indefinite
}

impl MatrixDefiniteness {
    fn to_mkl_opt(&self) -> MklInt {
        use MatrixDefiniteness::*;
        match self {
            PositiveDefinite => MKL_DSS_POSITIVE_DEFINITE as MklInt,
            Indefinite => MKL_DSS_INDEFINITE as MklInt,
        }
    }
}



pub fn check_csr(row_ptr: &[MklInt], _columns: &[MklInt]) {
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
    marker: PhantomData<T>,
    num_rows: usize,
    nnz: usize
}

impl<T> SymbolicFactorization<T>
where
    T: SupportedScalar
{
    pub fn try_from_csr(row_ptr: &[MklInt], columns: &[MklInt], structure: MatrixStructure) -> Result<Self, DssError> {
        check_csr(row_ptr, columns);

        // TODO: Result instead of panic?
        assert!(row_ptr.len() > 0);

        let create_opts = (MKL_DSS_DEFAULTS + MKL_DSS_ZERO_BASED_INDEXING) as MklInt;
        let mut handle = Handle::create(create_opts)?;

        let define_opts = structure.to_mkl_opt();

        let nnz = columns.len();
        let num_rows = row_ptr.len() - 1;
        let num_cols = num_rows;

        unsafe {
            // TODO: Handle errors
            let error = dss_define_structure_(&mut handle.handle,
                                  &define_opts,
                                  row_ptr.as_ptr(),
                                  &(num_rows as MklInt),
                                  &(num_cols as MklInt),
                                  columns.as_ptr(),
                                  &(nnz as MklInt)
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

        Ok(Self {
            handle,
            marker: PhantomData,
            num_rows,
            nnz
        })
    }

    pub fn factor(self, values: &[T], definiteness: MatrixDefiniteness) -> NumericalFactorization<T> {
        let mut factorization = NumericalFactorization {
            handle: self.handle,
            marker: PhantomData,
            num_rows: self.num_rows,
            nnz: self.nnz
        };
        // TODO: Return proper error
        factorization.refactor(values, definiteness);
        factorization
    }
}

pub struct NumericalFactorization<T> {
    handle: Handle,
    marker: PhantomData<T>,
    num_rows: usize,
    nnz: usize
}

impl<T> NumericalFactorization<T>
where
    T: SupportedScalar
{
    pub fn refactor(&mut self, values: &[T], definiteness: MatrixDefiniteness) {
        // TODO: Part of error?
        assert_eq!(values.len(), self.nnz);

        let opts = definiteness.to_mkl_opt();

        // TODO: Must save e.g. size of sparsity pattern earlier in the process
        // so that we may verify that the length of values is correct.
        // Otherwise we may invoke UB.

        unsafe {
            // TODO: Handle errors
            let error = dss_factor_real_(&mut self.handle.handle,
                                         &opts,
                                         values.as_ptr() as *const c_void);
            if error != 0 {
                eprintln!("dss_factor_real_ error: {}", error);
            }
        }
    }

    // TODO: Would it be safe to only take &self and still hand in a mutable pointer
    // to the handle? We technically don't have any idea what is happening inside
    // MKL, but on the other hand the factorization cannot be accessed from multiple threads,
    // and I think as far as I can tell that the state of the factorization does not change?
    // Unless an error somehow invalidates the handle? Not clear...
    // Note: same for diagonal/backward
    pub fn forward_substitute_into(&mut self, solution: &mut [T], rhs: &[T]) {
        let num_rhs = rhs.len() / self.num_rows;

        // TODO: Make part of error?
        assert_eq!(rhs.len() % self.num_rows, 0,
                   "Number of entries in RHS must be divisible by system size.");
        assert_eq!(solution.len(), rhs.len());


        // TODO: Error handling
        let error = unsafe { dss_solve_real_(&mut self.handle.handle,
                                    &(MKL_DSS_FORWARD_SOLVE as MklInt),
                                    rhs.as_ptr() as *const c_void,
                                    &(num_rhs as MklInt),
                                    solution.as_mut_ptr() as *mut c_void) };
        if error != 0 {
            eprintln!("dss_factor_real_ error (forward): {}", error);
        }
    }

    pub fn diagonal_substitute_into(&mut self, solution: &mut [T], rhs: &[T]) {
        let num_rhs = rhs.len() / self.num_rows;

        // TODO: Make part of error?
        assert_eq!(rhs.len() % self.num_rows, 0,
                   "Number of entries in RHS must be divisible by system size.");
        assert_eq!(solution.len(), rhs.len());

        let error = unsafe { dss_solve_real_(&mut self.handle.handle,
                                    &(MKL_DSS_DIAGONAL_SOLVE as MklInt),
                                    rhs.as_ptr() as *const c_void,
                                    &(num_rhs as MklInt),
                                    solution.as_mut_ptr() as *mut c_void) };

        if error != 0 {
            eprintln!("dss_factor_real_ error (diagonal): {}", error);
        }
    }

    pub fn backward_substitute_into(&mut self, solution: &mut [T], rhs: &[T]) {
        let num_rhs = rhs.len() / self.num_rows;

        // TODO: Make part of error?
        assert_eq!(rhs.len() % self.num_rows, 0,
                   "Number of entries in RHS must be divisible by system size.");
        assert_eq!(solution.len(), rhs.len());

        let error = unsafe { dss_solve_real_(&mut self.handle.handle,
                                    &(MKL_DSS_BACKWARD_SOLVE as MklInt),
                                    rhs.as_ptr() as *const c_void,
                                    &(num_rhs as MklInt),
                                    solution.as_mut_ptr() as *mut c_void) };

        if error != 0 {
            eprintln!("dss_factor_real_ error (backward): {}", error);
        }
    }

    /// Convenience function for calling the different substitution phases.
    ///
    /// `buffer` must have same size as `solution`.
    pub fn solve_into(&mut self, solution: &mut [T], buffer: &mut [T], rhs: &[T]) {
        let y = solution;
        self.forward_substitute_into(y, rhs);

        let z = buffer;
        self.diagonal_substitute_into(z, &y);

        let x = y;
        self.backward_substitute_into(x, &z);
    }
}