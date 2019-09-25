use crate::sparse::{CsrMatrixHandle, MatrixDescription, SparseStatusError};
use crate::util::is_same_type;
use crate::SupportedScalar;

use mkl_sys::{mkl_sparse_d_ev, mkl_sparse_ee_init, MKL_INT};

pub struct EigenResult<T> {
    eigenvectors: Vec<T>,
    eigenvalues: Vec<T>,
    residuals: Vec<T>,
}

impl<T> EigenResult<T> {
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> &[T] {
        &self.eigenvectors
    }

    pub fn residuals(&self) -> &[T] {
        &self.residuals
    }
}

#[derive(Debug, PartialEq, Eq)]
enum WhichType {
    Largest,
    Smallest,
}

fn extremal_eigenvalues<T>(
    which: WhichType,
    matrix: &CsrMatrixHandle<T>,
    description: &MatrixDescription,
    k: usize,
) -> Result<EigenResult<T>, SparseStatusError>
where
    T: SupportedScalar,
{
    let k_in = k as MKL_INT;
    let mut k_out = 0 as MKL_INT;

    if is_same_type::<T, f64>() {
        // TODO: Allow tweaking options
        let mut opts = vec![0 as MKL_INT; 128];
        let code = unsafe { mkl_sparse_ee_init(opts.as_mut_ptr()) };
        SparseStatusError::new_result(code, "mkl_sparse_ee_init")?;

        let mut eigenvalues = vec![T::zero_element(); k];
        let mut eigenvectors = vec![T::zero_element(); k * matrix.cols()];
        let mut residuals = vec![T::zero_element(); k];

        let mut which = match which {
            WhichType::Largest => 'L' as i8,
            WhichType::Smallest => 'S' as i8,
        };

        let code = unsafe {
            mkl_sparse_d_ev(
                &mut which,
                opts.as_mut_ptr(),
                matrix.handle,
                description.to_mkl_descr(),
                k_in,
                &mut k_out,
                eigenvalues.as_mut_ptr() as *mut f64,
                eigenvectors.as_mut_ptr() as *mut f64,
                residuals.as_mut_ptr() as *mut f64,
            )
        };
        SparseStatusError::new_result(code, "mkl_sparse_d_ev")?;
        let k_out = k_out as usize;
        eigenvalues.truncate(k_out);
        eigenvectors.truncate(k_out);
        residuals.truncate(k_out);
        Ok(EigenResult {
            eigenvectors,
            eigenvalues,
            residuals,
        })
    } else {
        panic!("Unsupported type");
    }
}

/// Attempts to compute the `k` largest eigenvalues of the given matrix, with the given description.
///
/// Note that the returned number of eigenvalues might be smaller than requested (see MKL
/// docs for details).
pub fn k_largest_eigenvalues<T>(
    matrix: &CsrMatrixHandle<T>,
    description: &MatrixDescription,
    k: usize,
) -> Result<EigenResult<T>, SparseStatusError>
where
    T: SupportedScalar,
{
    extremal_eigenvalues(WhichType::Largest, matrix, description, k)
}

/// Attempts to compute the `k` smallest eigenvalues of the given matrix, with the given description.
///
/// Note that the returned number of eigenvalues might be smaller than requested (see MKL
/// docs for details).
// TODO: Extend to general sparse matrices, not just CSR
pub fn k_smallest_eigenvalues<T>(
    matrix: &CsrMatrixHandle<T>,
    description: &MatrixDescription,
    k: usize,
) -> Result<EigenResult<T>, SparseStatusError>
where
    T: SupportedScalar,
{
    extremal_eigenvalues(WhichType::Smallest, matrix, description, k)
}