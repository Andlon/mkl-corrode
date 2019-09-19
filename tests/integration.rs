use mkl_corrode::dss::{Definiteness, MatrixStructure, Solver};

use approx::assert_abs_diff_eq;

use MatrixStructure::Symmetric;
use Definiteness::PositiveDefinite;
use mkl_corrode::dss::MatrixStructure::NonSymmetric;
use mkl_corrode::dss::Definiteness::Indefinite;

#[test]
fn dss_1x1_factorization() {
    let row_ptr = [0, 1];
    let columns = [0];
    let values = [2.0];

    let mut fact = Solver::try_factor(&row_ptr, &columns, &values, Symmetric, PositiveDefinite)
        .unwrap();

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

    let mut fact = Solver::try_factor(&row_ptr, &columns, &values, NonSymmetric, Indefinite)
        .unwrap();

    let rhs = [7.0, -13.0, 2.0, -1.0];
    let mut sol = [0.0, 0.0, 0.0, 0.0];
    let mut buffer = sol.clone();
    fact.solve_into(&mut sol, &mut buffer, &rhs).unwrap();
    let expected_sol = [-(1.0 / 3.0), -2.0, 5.0 / 3.0, 1.0];

    assert_abs_diff_eq!(sol.as_ref(), expected_sol.as_ref(), epsilon = 1e-12);
}

#[test]
fn dss_symmetric_factorization() {
    // Matrix
    // [10, 0, 2,
    //   0, 5, 1
    //   2  1  4]
    let row_ptr = [0, 2, 4, 7];
    let columns = [0, 2, 1, 2, 0, 1, 2];
    let values = [10.0, 2.0, 5.0, 1.0, 2.0, 1.0, 4.0];

    let mut fact = Solver::try_factor(&row_ptr, &columns, &values, NonSymmetric, Indefinite)
        .unwrap();

    let rhs = [2.0, -3.0, 5.0];
    let solution = fact.solve(&rhs).unwrap();
    let expected_sol = [-0.10588235, -0.90588235,  1.52941176];

    assert_abs_diff_eq!(solution.as_ref(), expected_sol.as_ref(), epsilon = 1e-6);
}
