use mkl_corrode::dss::{SymbolicFactorization, MatrixStructure, MatrixDefiniteness};

use approx::assert_abs_diff_eq;

#[test]
fn dss_symbolic_factorization() {
    let row_ptr = [
        0, 1
    ];
    let columns = [
        0
    ];

    let _ = SymbolicFactorization::<f64>::from_csr(&row_ptr, &columns, MatrixStructure::NonSymmetric);
}

#[test]
fn dss_1x1_factorization() {
    let row_ptr = [0, 1];
    let columns = [0];
    let values = [2.0];

    let symb = SymbolicFactorization::from_csr(&row_ptr, &columns, MatrixStructure::Symmetric);
    let mut fact = symb.factor(&values, MatrixDefiniteness::PositiveDefinite);

    let rhs = [2.0];
    let mut sol = [0.0];
    let mut buffer = [0.0];
    fact.solve_into(&mut sol, &mut buffer, &rhs);

    let expected_sol = [1.0];
    assert_abs_diff_eq!(sol.as_ref(), expected_sol.as_ref(), epsilon=1e-12);
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

    let symb = SymbolicFactorization::from_csr(&row_ptr, &columns, MatrixStructure::NonSymmetric);
    let mut fact = symb.factor(&values, MatrixDefiniteness::Indefinite);

    let rhs = [7.0, -13.0, 2.0, -1.0];
    let mut sol = [0.0, 0.0, 0.0, 0.0];
    let mut buffer = sol.clone();
    fact.solve_into(&mut sol, &mut buffer, &rhs);
    let expected_sol = [-(1.0/3.0), -2.0, 5.0/3.0, 1.0];

    assert_abs_diff_eq!(sol.as_ref(), expected_sol.as_ref(), epsilon=1e-12);
}