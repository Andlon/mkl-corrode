use mkl_sys::{
    _MKL_DSS_HANDLE_t, dss_create_, dss_define_structure_, dss_delete_, dss_factor_real_,
    dss_reorder_, dss_solve_real_, dss_statistics_, MKL_DSS_AUTO_ORDER, MKL_DSS_BACKWARD_SOLVE,
    MKL_DSS_DEFAULTS, MKL_DSS_DIAGONAL_SOLVE, MKL_DSS_FORWARD_SOLVE, MKL_DSS_INDEFINITE,
    MKL_DSS_METIS_OPENMP_ORDER, MKL_DSS_POSITIVE_DEFINITE, MKL_DSS_ZERO_BASED_INDEXING, MKL_INT,
};
use std::ffi::{c_void, CStr};
use std::marker::PhantomData;
use std::ptr::{null, null_mut};

// MKL constants
use crate::dss::SparseMatrix;
use crate::SupportedScalar;
use core::fmt;
use mkl_sys::{
    MKL_DSS_COL_ERR, MKL_DSS_DIAG_ERR, MKL_DSS_FAILURE, MKL_DSS_I32BIT_ERR, MKL_DSS_INVALID_OPTION,
    MKL_DSS_MSG_LVL_ERR, MKL_DSS_MSG_LVL_ERROR, MKL_DSS_MSG_LVL_FATAL, MKL_DSS_MSG_LVL_INFO,
    MKL_DSS_MSG_LVL_WARNING, MKL_DSS_NOT_SQUARE, MKL_DSS_OOC_MEM_ERR, MKL_DSS_OOC_OC_ERR,
    MKL_DSS_OOC_RW_ERR, MKL_DSS_OPTION_CONFLICT, MKL_DSS_OUT_OF_MEMORY, MKL_DSS_REORDER1_ERR,
    MKL_DSS_REORDER_ERR, MKL_DSS_ROW_ERR, MKL_DSS_STATE_ERR, MKL_DSS_STATISTICS_INVALID_MATRIX,
    MKL_DSS_STATISTICS_INVALID_STATE, MKL_DSS_STATISTICS_INVALID_STRING, MKL_DSS_STRUCTURE_ERR,
    MKL_DSS_SUCCESS, MKL_DSS_TERM_LVL_ERR, MKL_DSS_TOO_FEW_VALUES, MKL_DSS_TOO_MANY_VALUES,
    MKL_DSS_VALUES_ERR,
};
use std::fmt::{Debug, Display, Formatter};

/// Calls the given DSS function, noting its error code and upon a non-success result,
/// returns an appropriate error.
macro_rules! dss_call {
    ($routine:ident ($($arg: tt)*)) => {
        {
            let result: Result<(), Error> = {
                let code = $routine($($arg)*);
                if code == MKL_DSS_SUCCESS {
                    Ok(())
                } else {
                    Err(Error::new(ErrorCode::from_return_code(code), stringify!($routine)))
                }
            };
            result
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Error {
    code: ErrorCode,
    routine: &'static str,
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
        write!(
            f,
            "Error in routine {}. Return code: {:?}",
            self.routine(),
            self.return_code()
        )
    }
}

impl std::error::Error for Error {}

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
        unsafe {
            dss_call! { dss_create_(&mut handle, &options) }
        }?;
        Ok(Self { handle })
    }

    fn get_statistics(&mut self) -> SolverStatistics {
        let reorder_str = CStr::from_bytes_with_nul(b"ReorderTime\0").unwrap();
        let factor_str = CStr::from_bytes_with_nul(b"FactorTime\0").unwrap();
        let solve_str = CStr::from_bytes_with_nul(b"SolveTime\0").unwrap();

        // SAFETY: The output buffer must be large enough to accommodate outputs for all
        // options that we require. This has to be checked with MKL docs, since
        // it might write more info to the buffer than we need right now.
        // For now we use a much too large buffer to hopefully help avoid any undefined behavior
        // due to, for example, MKL adding additional outputs for the same strings in the future
        // (that would be terrible, but given the state of the MKL library I wouldn't put
        // it past them).
        let mut output = [0.0f64; 64];
        SolverStatistics {
            reorder_time: unsafe {
                dss_call!(dss_statistics_(
                    &mut self.handle,
                    &MKL_DSS_DEFAULTS,
                    reorder_str.as_ptr(),
                    output.as_mut_ptr()
                ))
            }
            .ok()
            .map(|_| output[0]),
            factor_time: unsafe {
                dss_call!(dss_statistics_(
                    &mut self.handle,
                    &MKL_DSS_DEFAULTS,
                    factor_str.as_ptr(),
                    output.as_mut_ptr()
                ))
            }
            .ok()
            .map(|_| output[0]),
            solve_time: unsafe {
                dss_call!(dss_statistics_(
                    &mut self.handle,
                    &MKL_DSS_DEFAULTS,
                    solve_str.as_ptr(),
                    output.as_mut_ptr()
                ))
            }
            .ok()
            .map(|_| output[0]),
        }
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

#[derive(Clone, Debug)]
#[non_exhaustive]
/// Statistics, such as timing.
///
/// In general, statistics are only provided for completed phases, in which case the statistic
/// corresponding to the most recent execution of the phase is given.
pub struct SolverStatistics {
    pub reorder_time: Option<f64>,
    pub factor_time: Option<f64>,
    pub solve_time: Option<f64>,
    // determinant_time: Option<f64>,
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

/// The message log level for the DSS solver.
#[derive(Debug, Clone)]
pub enum MessageLevel {
    Info,
    Warning,
    Error,
    Fatal,
}

impl MessageLevel {
    fn to_mkl_int(&self) -> MKL_INT {
        match self {
            MessageLevel::Info => MKL_DSS_MSG_LVL_INFO,
            MessageLevel::Warning => MKL_DSS_MSG_LVL_WARNING,
            MessageLevel::Error => MKL_DSS_MSG_LVL_ERROR,
            MessageLevel::Fatal => MKL_DSS_MSG_LVL_FATAL,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolverOptions {
    parallel_reorder: bool,
    message_level: MessageLevel,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            parallel_reorder: false,
            message_level: MessageLevel::Fatal,
        }
    }
}

impl SolverOptions {
    pub fn parallel_reorder(self, enable: bool) -> Self {
        Self {
            parallel_reorder: enable,
            ..self
        }
    }

    /// Sets the message level of the solver.
    ///
    /// By default, only "Fatal" messages will be printed to stdout/stderr. Increasing the
    /// message level may cause the DSS solver to produce more output.
    pub fn message_level(self, message_level: MessageLevel) -> Self {
        Self {
            message_level,
            ..self
        }
    }
}

/// An error returned by [`Solver`].
#[derive(Debug)]
pub enum SolverError {
    /// An error occurred while defining the structure of the matrix.
    DefineStructure(Error),
    /// An error occured during symbolic factorization / reordering.
    Reorder(Error),
    /// An error occured during numerical factorization.
    Factor(Error),
    /// An error occured during a solve stage.
    Solve(Error),
    /// An error occured in an MKL function not covered by the other variants.
    OtherMklRoutine(Error),
}

impl Display for SolverError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::DefineStructure(err) => write!(
                f,
                "An error occurred while defining the matrix structure. Error: {err}"
            ),
            SolverError::Reorder(err) => write!(
                f,
                "An error occurred during symbolic factorization. Error: {err}"
            ),
            SolverError::Factor(err) => write!(
                f,
                "An error occurred during numerical factorization. Error: {err}"
            ),
            SolverError::Solve(err) => {
                write!(f, "An error occurred during the solve phase. Error: {err}")
            }
            SolverError::OtherMklRoutine(err) => {
                write!(f, "An error occurred in an MKL routine. Error: {err}")
            }
        }
    }
}

impl std::error::Error for SolverError {
    fn cause(&self) -> Option<&dyn std::error::Error> {
        match self {
            SolverError::DefineStructure(err)
            | SolverError::Reorder(err)
            | SolverError::Factor(err)
            | SolverError::Solve(err)
            | SolverError::OtherMklRoutine(err) => Some(err),
        }
    }
}

pub struct Solver<T> {
    handle: Handle,
    marker: PhantomData<T>,
    num_rows: usize,
    nnz: usize,
}

impl<T> Debug for Solver<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct(std::any::type_name::<Self>())
            .field("handle", &"<n/a>")
            .field("num_rows", &self.num_rows)
            .field("nnz", &self.nnz)
            .finish()
    }
}

impl<T> Solver<T>
where
    T: SupportedScalar,
{
    pub fn try_factor_with_opts(
        matrix: &SparseMatrix<T>,
        definiteness: Definiteness,
        options: &SolverOptions,
    ) -> Result<Self, SolverError> {
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

        let create_opts =
            MKL_DSS_DEFAULTS + MKL_DSS_ZERO_BASED_INDEXING + options.message_level.to_mkl_int();
        let mut handle = Handle::create(create_opts).map_err(SolverError::OtherMklRoutine)?;

        let define_opts = structure.to_mkl_opt();
        unsafe {
            dss_call! { dss_define_structure_(
                    &mut handle.handle,
                    &define_opts,
                    row_ptr.as_ptr(),
                    // TODO: What if num_rows, nnz or num_cols > max(MKL_INT)?
                    &(num_rows as MKL_INT),
                    &(num_cols as MKL_INT),
                    columns.as_ptr(),
                    &(nnz as MKL_INT),
            ) }
        }
        .map_err(SolverError::DefineStructure)?;

        let reorder_opts;
        if options.parallel_reorder {
            reorder_opts = MKL_DSS_METIS_OPENMP_ORDER;
        } else {
            reorder_opts = MKL_DSS_AUTO_ORDER;
        }
        unsafe {
            dss_call! { dss_reorder_(&mut handle.handle, &reorder_opts, null()) }
        }
        .map_err(SolverError::Reorder)?;

        let mut factorization = Solver {
            handle,
            num_rows,
            nnz,
            marker: PhantomData,
        };
        factorization.refactor(values, definiteness)?;
        Ok(factorization)
    }

    /// Factors with default options.
    pub fn try_factor(
        matrix: &SparseMatrix<T>,
        definiteness: Definiteness,
    ) -> Result<Self, SolverError> {
        Self::try_factor_with_opts(matrix, definiteness, &SolverOptions::default())
    }

    pub fn refactor(
        &mut self,
        values: &[T],
        definiteness: Definiteness,
    ) -> Result<(), SolverError> {
        // TODO: Part of error?
        assert_eq!(values.len(), self.nnz);

        let opts = definiteness.to_mkl_opt();
        unsafe {
            dss_call! { dss_factor_real_(
                &mut self.handle.handle,
                &opts,
                values.as_ptr() as *const c_void,
            ) }
        }
        .map_err(SolverError::Factor)?;
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

        unsafe {
            dss_call! {
                dss_solve_real_(
                    &mut self.handle.handle,
                    &(MKL_DSS_FORWARD_SOLVE),
                    rhs.as_ptr() as *const c_void,
                    // TODO: What if num_rhs > max(MKL_INT)? Absurd situation, but it could maybe
                    // lead to undefined behavior, so we need to handle it
                    &(num_rhs as MKL_INT),
                    solution.as_mut_ptr() as *mut c_void,
                )
            }
        }?;
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

        unsafe {
            dss_call! {
                dss_solve_real_(
                    &mut self.handle.handle,
                    &(MKL_DSS_DIAGONAL_SOLVE),
                    rhs.as_ptr() as *const c_void,
                    // TODO: See other comment about this coercion cast
                    &(num_rhs as MKL_INT),
                    solution.as_mut_ptr() as *mut c_void,
                )
            }
        }?;
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

        unsafe {
            dss_call! {
                dss_solve_real_(
                    &mut self.handle.handle,
                    &(MKL_DSS_BACKWARD_SOLVE),
                    rhs.as_ptr() as *const c_void,
                    // TODO: See other comment about num_rhs and `as` cast
                    &(num_rhs as MKL_INT),
                    solution.as_mut_ptr() as *mut c_void,
                )
            }
        }?;
        Ok(())
    }

    /// Convenience function for calling the different substitution phases.
    ///
    /// `buffer` must have same size as `solution`.
    pub fn solve_into(
        &mut self,
        solution: &mut [T],
        buffer: &mut [T],
        rhs: &[T],
    ) -> Result<(), Error> {
        let y = solution;
        self.forward_substitute_into(y, rhs)?;

        let z = buffer;
        self.diagonal_substitute_into(z, &y)?;

        let x = y;
        self.backward_substitute_into(x, &z)?;

        Ok(())
    }

    /// Convenience function that internally allocates buffer storage and output storage.
    pub fn solve(&mut self, rhs: &[T]) -> Result<Vec<T>, Error> {
        let mut solution = vec![T::zero_element(); rhs.len()];
        let mut buffer = vec![T::zero_element(); rhs.len()];
        self.solve_into(&mut solution, &mut buffer, rhs)?;
        Ok(solution)
    }

    /// Return statistics on the solver phases.
    ///
    /// TODO: Currently this is kinda broken. According to Intel docs, we should be able to
    /// request statistics for phases that we have not yet entered, in which case we'd return
    /// an error. Unfortunately, this is *not* the case. Instead, MKL happily returns a
    /// "success" error code, and instead prints a fatal error to the terminal.
    /// The result is that results of stages that not yet been completed might be completely wrong,
    /// and instead identical to whatever previous stage did *not* fail.
    pub fn get_statistics(&mut self) -> SolverStatistics {
        self.handle.get_statistics()
    }
}
