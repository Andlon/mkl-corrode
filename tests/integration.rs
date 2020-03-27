use mkl_corrode::dss::{Definiteness, MatrixStructure, Solver, SparseMatrix};

use approx::assert_abs_diff_eq;

use mkl_corrode::dss::Definiteness::Indefinite;
use mkl_corrode::dss::MatrixStructure::NonSymmetric;
use mkl_corrode::extended_eigensolver::{k_largest_eigenvalues, k_smallest_eigenvalues};
use mkl_corrode::sparse::{CsrMatrixHandle, MatrixDescription, SparseMatrixType};
use Definiteness::PositiveDefinite;
use MatrixStructure::Symmetric;

#[test]
fn dss_1x1_factorization() {
    let row_ptr = [0, 1];
    let columns = [0];
    let values = [2.0];

    let matrix =
        SparseMatrix::try_convert_from_csr(&row_ptr, &columns, &values, Symmetric).unwrap();
    let mut fact = Solver::try_factor(&matrix, PositiveDefinite).unwrap();

    let rhs = [2.0];
    let mut sol = [0.0];
    let mut buffer = [0.0];
    fact.solve_into(&mut sol, &mut buffer, &rhs).unwrap();

    let expected_sol = [1.0];
    assert_abs_diff_eq!(sol.as_ref(), expected_sol.as_ref(), epsilon = 1e-12);
}

#[test]
fn dss_factorization() {
    // Matrix:
    // [10, 0, 2, 7,
    //   3, 6, 0, 0,
    //   0, 7, 9, 1,
    //   0, 2, 0, 3]

    let row_ptr = [0, 3, 5, 8, 10];
    let columns = [0, 2, 3, 0, 1, 1, 2, 3, 1, 3];
    let values = [10.0, 2.0, 7.0, 3.0, 6.0, 7.0, 9.0, 1.0, 2.0, 3.0];

    let matrix =
        SparseMatrix::try_convert_from_csr(&row_ptr, &columns, &values, NonSymmetric).unwrap();
    let mut fact = Solver::try_factor(&matrix, Indefinite).unwrap();

    let rhs = [7.0, -13.0, 2.0, -1.0];
    let mut sol = [0.0, 0.0, 0.0, 0.0];
    let mut buffer = sol.clone();
    fact.solve_into(&mut sol, &mut buffer, &rhs).unwrap();
    let expected_sol = [-(1.0 / 3.0), -2.0, 5.0 / 3.0, 1.0];

    assert_abs_diff_eq!(sol.as_ref(), expected_sol.as_ref(), epsilon = 1e-12);
}

#[test]
fn dss_symmetric_posdef_factorization() {
    // Redundantly stored entries (i.e. lower triangular portion explicitly stored
    {
        // Matrix
        // [10, 0, 2,
        //   0, 5, 1
        //   2  1  4]
        let row_ptr = [0, 2, 4, 7];
        let columns = [0, 2, 1, 2, 0, 1, 2];
        let values = [10.0, 2.0, 5.0, 1.0, 2.0, 1.0, 4.0];

        let matrix =
            SparseMatrix::try_convert_from_csr(&row_ptr, &columns, &values, Symmetric).unwrap();
        let mut fact = Solver::try_factor(&matrix, PositiveDefinite).unwrap();

        let rhs = [2.0, -3.0, 5.0];
        let solution = fact.solve(&rhs).unwrap();
        let expected_sol = [-0.10588235, -0.90588235, 1.52941176];

        assert_abs_diff_eq!(solution.as_ref(), expected_sol.as_ref(), epsilon = 1e-6);
    }

    // Same test, but store only upper triangular part of matrix
    {
        // Matrix
        // [10, 0, 2,
        //   0, 5, 1
        //   2  1  4]
        let row_ptr = [0, 2, 4, 5];
        let columns = [0, 2, 1, 2, 2];
        let values = [10.0, 2.0, 5.0, 1.0, 4.0];

        let matrix =
            SparseMatrix::try_convert_from_csr(&row_ptr, &columns, &values, Symmetric).unwrap();
        let mut fact = Solver::try_factor(&matrix, PositiveDefinite).unwrap();

        let rhs = [2.0, -3.0, 5.0];
        let solution = fact.solve(&rhs).unwrap();
        let expected_sol = [-0.10588235, -0.90588235, 1.52941176];

        assert_abs_diff_eq!(solution.as_ref(), expected_sol.as_ref(), epsilon = 1e-6);
    }
}

#[test]
fn csr_unsafe_construction_destruction() {
    // Matrix
    // [10, 0, 2,
    //   0, 5, 1
    //   2  1  4]
    let row_ptr = [0, 2, 4, 7];
    let columns = [0, 2, 1, 2, 0, 1, 2];
    let values = [10.0, 2.0, 5.0, 1.0, 2.0, 1.0, 4.0];

    let matrix = unsafe {
        CsrMatrixHandle::from_raw_csr_data(
            3,
            3,
            &row_ptr[..row_ptr.len() - 1],
            &row_ptr[1..],
            &columns,
            &values,
        )
    };
    drop(matrix);

    // Check that dropping the handle does not "destroy" the input data
    // (note: it may be necessary to run this test through Valgrind and/or adress/memory sanitizers
    // to make sure that it works as intended.
    assert_eq!(row_ptr, [0, 2, 4, 7]);
    assert_eq!(columns, [0, 2, 1, 2, 0, 1, 2]);
    assert_eq!(values, [10.0, 2.0, 5.0, 1.0, 2.0, 1.0, 4.0]);
}

#[test]
fn basic_k_smallest_largest_eigenvalues() {
    // Matrix
    // [10, 0, 2,
    //   0, 5, 1
    //   2  1  4]
    let row_ptr = [0, 2, 4, 7];
    let columns = [0, 2, 1, 2, 0, 1, 2];
    let values = [10.0, 2.0, 5.0, 1.0, 2.0, 1.0, 4.0];
    let matrix = unsafe {
        CsrMatrixHandle::from_raw_csr_data(
            3,
            3,
            &row_ptr[..row_ptr.len() - 1],
            &row_ptr[1..],
            &columns,
            &values,
        )
    }
    .unwrap();

    let description = MatrixDescription::default().with_type(SparseMatrixType::General);
    let expected_eigvals = vec![2.94606902, 5.43309508, 10.6208359];
    let largest1 = k_largest_eigenvalues(&matrix, &description, 1).unwrap();
    let largest2 = k_largest_eigenvalues(&matrix, &description, 2).unwrap();
    let largest3 = k_largest_eigenvalues(&matrix, &description, 3).unwrap();

    assert_abs_diff_eq!(
        largest1.eigenvalues(),
        &expected_eigvals[2..=2],
        epsilon = 1e-6
    );

    assert_abs_diff_eq!(
        largest2.eigenvalues(),
        &expected_eigvals[1..=2],
        epsilon = 1e-6
    );

    assert_abs_diff_eq!(
        largest3.eigenvalues(),
        &expected_eigvals[0..=2],
        epsilon = 1e-6
    );

    let smallest1 = k_smallest_eigenvalues(&matrix, &description, 1).unwrap();
    let smallest2 = k_smallest_eigenvalues(&matrix, &description, 2).unwrap();
    let smallest3 = k_smallest_eigenvalues(&matrix, &description, 3).unwrap();

    assert_abs_diff_eq!(
        smallest1.eigenvalues(),
        &expected_eigvals[0..=0],
        epsilon = 1e-6
    );

    assert_abs_diff_eq!(
        smallest2.eigenvalues(),
        &expected_eigvals[0..=1],
        epsilon = 1e-6
    );

    assert_abs_diff_eq!(
        smallest3.eigenvalues(),
        &expected_eigvals[0..=2],
        epsilon = 1e-6
    );
}

#[test]
fn dss_solver_debug() {
    use std::fmt::Write;

    let row_ptr = [0, 1];
    let columns = [0];
    let values = [0.0];

    // Construct dummy matrix
    let matrix = SparseMatrix::try_convert_from_csr(&row_ptr, &columns, &values, NonSymmetric)
        .unwrap();
    let solver = Solver::try_factor(&matrix, Indefinite).unwrap();

    let mut debug_str = String::new();
    write!(&mut debug_str, "{:?}", solver).unwrap();

    assert_eq!(debug_str,
               "mkl_corrode::dss::solver::Solver<f64> { handle: \"<n/a>\", num_rows: 1, nnz: 1 }");
}
