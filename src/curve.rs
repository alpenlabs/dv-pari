//! # curve.rs

#![allow(unexpected_cfgs)]
use ark_ff::PrimeField;
use ark_ff::fields::{Fp256, MontBackend, MontConfig};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::os::raw::c_void;
use xs233_sys::{xsk233_add, xsk233_generator, xsk233_neutral, xsk233_point};

/// FqConfig for sect233k1
#[derive(MontConfig, Debug)]
#[modulus = "3450873173395281893717377931138512760570940988862252126328087024741343"]
#[generator = "3"]
pub struct FqConfig;

/// Fr
pub type Fr = Fp256<MontBackend<FqConfig, 4>>;

/// CurvePoint
#[derive(Debug, Clone, Copy)]
pub struct CurvePoint(pub xsk233_point);

/// CompressedCurvePoint
pub type CompressedCurvePoint = [u8; 30];

impl PartialEq for CurvePoint {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let r = xs233_sys::xsk233_equals(&self.0, &other.0);
            r != 0
        }
    }
}
impl Eq for CurvePoint {}

impl CurvePoint {
    pub(crate) fn generator() -> Self {
        unsafe { CurvePoint(xsk233_generator) }
    }

    pub(crate) fn add(a: CurvePoint, b: CurvePoint) -> Self {
        unsafe {
            let mut p3 = xsk233_neutral;
            xsk233_add(&mut p3, &a.0, &b.0);
            CurvePoint(p3)
        }
    }

    pub(crate) fn to_bytes(self) -> [u8; 30] {
        unsafe {
            let pt = self.0;
            let mut dst = [0u8; 30];
            xs233_sys::xsk233_encode(dst.as_mut_ptr() as *mut c_void, &pt);
            dst
        }
    }

    pub(crate) fn from_bytes(src: &mut [u8; 30]) -> CurvePoint {
        unsafe {
            let mut pt2 = xsk233_neutral;
            let success = xs233_sys::xsk233_decode(&mut pt2, src.as_mut_ptr() as *mut c_void);
            assert!(success != 0);
            CurvePoint(pt2)
        }
    }
}

pub(crate) fn point_scalar_mul(scalar: Fr, point: CurvePoint) -> CurvePoint {
    let scalar = fr_to_le_bytes(&scalar);

    unsafe {
        let mut result = xsk233_neutral;
        xs233_sys::xsk233_mul_frob(
            &mut result,
            &point.0,
            scalar.as_ptr() as *const _,
            scalar.len(),
        );
        CurvePoint(result)
    }
}

/// Point Scalar Multiplication with [`generator`] as the [`CurvePoint`]
pub(crate) fn point_scalar_mul_gen(scalar: Fr) -> CurvePoint {
    let scalar = fr_to_le_bytes(&scalar);

    unsafe {
        let mut result = xsk233_neutral;
        xs233_sys::xsk233_mulgen_frob(&mut result, scalar.as_ptr() as *const _, scalar.len());
        CurvePoint(result)
    }
}

/// Multi Scalar Multiplication
/// For now we just compute individual point scalar multiplications
/// and sum up the result
pub(crate) fn multi_scalar_mul(scalars: &[Fr], points: &[CurvePoint]) -> CurvePoint {
    assert_eq!(scalars.len(), points.len());

    let results_par_iter = points
        .par_iter() // Use Rayon's parallel iterator for points
        .zip(scalars.par_iter()) // Use Rayon's parallel iterator for scalars
        .map(|(p, s)| point_scalar_mul(*s, *p))
        .into_par_iter();

    results_par_iter.reduce(
        || unsafe { CurvePoint(xsk233_neutral) },
        |p1: CurvePoint, p2: CurvePoint| unsafe {
            let mut p3 = xsk233_neutral;
            xsk233_add(&mut p3, &p1.0, &p2.0);
            CurvePoint(p3)
        },
    )
}

/// Convert scalar field element to byte array
/// Pornin's [`xsk233_mulgen_frob`] accepts scalar as a byte array
fn fr_to_le_bytes(fr: &Fr) -> Vec<u8> {
    let big_int = fr.into_bigint();
    let limbs = big_int.0;

    let mut bytes = Vec::with_capacity(32);
    for limb in limbs.iter() {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    bytes.truncate(30);

    // remove trailing zeros
    // helps reduce iteration in double-and-add iterations
    while let Some(&last) = bytes.last() {
        if last == 0 {
            bytes.pop();
        } else {
            break;
        }
    }
    bytes
}

#[cfg(test)]
mod unit_test {
    use std::os::raw::c_void;

    use ark_ff::{AdditiveGroup, UniformRand};
    use ark_std::rand::thread_rng;
    use xs233_sys::{xsk233_add, xsk233_equals, xsk233_generator, xsk233_neutral};

    use crate::curve::{CurvePoint, point_scalar_mul};

    use super::{Fr, multi_scalar_mul};

    #[test]
    fn test_validate_psm_with_point_add() {
        let mut rng = thread_rng();
        let k1 = Fr::rand(&mut rng);
        let k2 = Fr::rand(&mut rng);

        unsafe {
            let d = CurvePoint(xsk233_generator);
            let y1 = point_scalar_mul(k1, d);
            let y2 = point_scalar_mul(k2, d);
            let y3 = point_scalar_mul(k1 + k2, d);

            let mut y12 = xsk233_neutral;
            xsk233_add(&mut y12, &y2.0, &y1.0);

            let is_iden = xsk233_equals(&y12, &y3.0);
            assert!(is_iden != 0);
        }
    }

    #[test]
    fn test_msm() {
        let mut rng = thread_rng();
        let n = 100_000;
        unsafe {
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let points: Vec<CurvePoint> = (0..n).map(|_| CurvePoint(xsk233_generator)).collect();
            let res = multi_scalar_mul(&scalars, &points);
            let mut total = Fr::ZERO;
            for scalar in scalars {
                total += scalar;
            }
            let total_msm = point_scalar_mul(total, points[0]);
            assert_eq!(total_msm, res);
        }
    }

    #[test]
    // Verifies that a CurvePoint is recovered after serialize-then-deserialize
    fn test_curve_point_to_bytes() {
        unsafe {
            let pt = xsk233_generator;
            let mut dst = [0u8; 30];
            xs233_sys::xsk233_encode(dst.as_mut_ptr() as *mut c_void, &pt);

            let mut pt2 = xsk233_neutral;
            let success = xs233_sys::xsk233_decode(&mut pt2, dst.as_mut_ptr() as *mut c_void);
            assert!(success != 0);
            let success = xsk233_equals(&pt2, &pt);
            assert!(success != 0);
        }
    }
}
