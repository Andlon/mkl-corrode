use mkl_corrode::dss::{SymbolicFactorization, MatrixStructure, MatrixDefiniteness};

#[test]
fn dss_symbolic_factorization() {
    let row_ptr = [
        0, 1
    ];
    let columns = [
        0
    ];

    let symb = SymbolicFactorization::<f64>::from_csr(&row_ptr, &columns, MatrixStructure::NonSymmetric);
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

    let symb = SymbolicFactorization::<f64>::from_csr(&row_ptr, &columns, MatrixStructure::NonSymmetric);
    let fact = symb.factor(&values, MatrixDefiniteness::Indefinite);
}