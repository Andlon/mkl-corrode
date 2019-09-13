use mkl_corrode::dss::{SymbolicFactorization, Structure};

#[test]
fn dss_symbolic_factorization() {
    let row_ptr = [
        0, 1
    ];
    let columns = [
        0
    ];

    let symb = SymbolicFactorization::<f64>::from_csr(&row_ptr, &columns, Structure::NonSymmetric);
}