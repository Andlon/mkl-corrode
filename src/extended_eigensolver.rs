use crate::sparse::{CsrMatrixHandle, SparseStatusError, MatrixDescription};
use crate::{SupportedScalar};
use crate::util::is_same_type;

use mkl_sys::{mkl_sparse_d_ev, mkl_sparse_ee_init, MKL_INT};

pub struct EigenResult<T>
{
    eigenvectors: Vec<T>,
    eigenvalues: Vec<T>,
    residuals: Vec<T>,
    k: usize
}

impl<T> EigenResult<T> {
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> &[T] {
        &self.eigenvectors
    }
}

pub fn k_largest_eigenvalues<T>(matrix: &CsrMatrixHandle<T>,
                                description: &MatrixDescription,
                                k: usize)
    -> Result<EigenResult<T>, SparseStatusError>
where
    T: SupportedScalar
{
    let k_in = k as MKL_INT;
    let mut k_out = 0 as MKL_INT;

    if is_same_type::<T, f64>() {
        // TODO: Allow tweaking options
        let mut opts = vec![0 as MKL_INT; 128];
        let code = unsafe { mkl_sparse_ee_init(opts.as_mut_ptr()) };
        SparseStatusError::new_result(code, "mkl_sparse_ee_init")?;

        let mut eigvals = vec![T::zero_element(); k];
        let mut eigvecs = vec![T::zero_element(); k * matrix.cols()];
        let mut res = vec![T::zero_element(); k];

        let mut which = 'L' as i8;
        let code = unsafe { mkl_sparse_d_ev(&mut which,
                                            opts.as_mut_ptr(),
                                            matrix.handle,
                                            description.to_mkl_descr(),
                                            k_in,
                                            &mut k_out,
                                            eigvals.as_mut_ptr() as *mut f64,
                                            eigvecs.as_mut_ptr() as *mut f64,
                                            res.as_mut_ptr() as *mut f64) };
        SparseStatusError::new_result(code, "mkl_sparse_d_ev")?;
        Ok(EigenResult {
            eigenvectors: eigvecs,
            eigenvalues: eigvals,
            residuals: res,
            k: k_out as usize
        })
    } else {
        panic!("Unsupported type");
    }
}