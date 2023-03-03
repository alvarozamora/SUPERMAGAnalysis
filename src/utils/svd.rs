use ndarray::{s, Array1, Array2};
use ndarray_linalg::{error::LinalgError, *};

/// Unlike the svd in `ndarray-linalg`, this computes the compact SVD and returns U, S, V
pub fn compact_svd<T: Lapack>(
    a: &Array2<T>,
    calc_u: bool,
    calc_vt: bool,
) -> Result<
    (
        Option<Array2<T>>,
        Array1<<T as Scalar>::Real>,
        Option<Array2<T>>,
    ),
    LinalgError,
> {
    let mut a = a.clone();

    let l = a.layout()?;
    let svd_res = T::svd(l, calc_u, calc_vt, a.as_allocated_mut()?)?;
    let (n, m) = l.size();

    // Calculate u if requested
    let u = svd_res.u.map(|u| {
        into_matrix(l.resized(n, n), u)
            .unwrap()
            .slice_move(s![.., ..n.min(m)])
    });

    // Calculate vt if requested
    let vt = svd_res.vt.map(|vt| {
        into_matrix(l.resized(m, m), vt)
            .unwrap()
            .slice_move(s![..n.min(m), ..])
    });

    // Vec to array
    let s = Array1::from(svd_res.s);

    Ok((u, s, vt))
}

#[test]
fn test_fifteen_by_three() {
    let a: Array2<f32> = Array2::from_shape_vec((15, 3), vec![1.0_f32; 15 * 3]).unwrap();

    let calc_u = true;
    let calc_vt = true;
    let (Some(u), s, Some(vt)) = compact_svd(&a, calc_u, calc_vt).unwrap() else { unreachable!() };
    assert_eq!(u.shape(), &[15, 3]);
    assert_eq!(s.shape(), &[3]);
    assert_eq!(vt.shape(), &[3, 3]);

    let us = u.dot(&Array2::from_diag(&s));
    println!("got us");
    let usvt = us.dot(&vt);
    println!("got usvt");

    for (ai_, ai) in usvt.into_iter().zip(a) {
        assert_rclose!(ai_, ai, 1e-5);
    }
}
