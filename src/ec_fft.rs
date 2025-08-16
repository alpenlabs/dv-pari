//! Utilities for ec-fft operations required by DV-Pari.
//!
//! We need a domain of sufficient length where we can do polynomial operations efficiently
//! For our purposes we need domain D and D' each of length `m` and a domain D U D' of length 2m,
//! where m is the `next power of 2` of the number of constraints
//! We need the following features :
//! # Polynomial Evaluation over the entire domain
//! # Generate vanishing polynomial coefficients
//! # Extend evaluations between domains D and D'
//!
//! Background on extend() function provided by ecfft library:
//! For the last use case [`FFtree::extend()`], we require a single FFTree over the combined domain
//! (D U D') to extend evaluations from domain D to D' i.e. evals_over_domain(D') <- FFtree::extend(evals_over_domain(D))
//! If we assume {d_i} be the elements in domain D and {d_i'} be the elements in domain D', then
//! the leaves of the combined tree is the combined domain D U D' and its leaves are arranged such that {d_i} and {d_i'} interleave each other
//! i.e. {d_j} = {d_i[0], d_i'[0], d_i[1], d_i'[1], ...., d_i[m-1], d_i'[m-1]}
//! Proof generation only requires a call to extend(), therefore we only initialize an instance of this tree.
//! FFtree data structure includes fields that are not essential for extend function. An FFTree instance that does not include
//! those fields is called a `minimal` tree
//!
//! Background on subtree_with_size() function provided by ecfft library:
//! [`FFtree::subtree_with_size(usize)`] function returns a subtree with leaves at even indices
//! For example, calling subtree_with_size(m) over combined tree (D U D' domain)  will return an FFTree over domain D
//! We need FFTree over domains D and D' during srs-setup time to calculate lagrange evaluations over \tau, and also during proof generation
//! to compute barycentric. So if a user is already in possession of a combined tree he can cheaply obtain subtrees with a call to the above function.
//!
//! Notes on size of FFTree:
//! The FFTree structure includes more than the domain. It also includes precomputed values useful for polynomial operations. Together this makes the size
//! of an instance of FFTree for our requirements quite large. 7.5 GB for domain size of 2^23 (D & D') and 15 GB for domain size of 2^24 (D U D').
//! To reduce memory pressure, we load the tree from disk only when it's needed and clear it afterwards. Additionally we use `minimal` tree, where possible.
//! A 'minimal' tree only includes fields necessary for specific use case (extend() during proof generation), in turn halving the memory usage requirement.
//!
//! Notes on different instantiations of FFTree:
//! We need operations on domain D, D' and D U D'
//! Over D: polynomial evaluation, lagrange evaluation over tau, barycentric weights, vanishing polynomial coefficients
//! Over D': lagrange over tau, extend
//! To generate vanishing polynomial over D', we need an FFTree of size 2m and with D' as its first subtree.
//! To generate vanishing polynomial over D and to extend evaluations from D to D', we need and FFtree of size 2m with domain D as its first subtree
//!
//! Notes on parameters for instantiations of different FFTrees:
//! Ecfft works over additive subgroup. We first find subgroup_generator, subgroup_order, coset_offset for FFTree over our field (sect233k1 scalar field).
//! To instantiate the specific types of FFTree: Tree2n, Tree2nd <- FFtree over combined domain with first subtree D and D' respectively, we consider the following:
//! we first find subgroup_generator for domain size '2m' which we call `base_generator`.
//!
//! FFtree with leaves given by coset + [0,1..,2m-1]*G gives Tree2n
//! FFtree with leaves given by (coset+G) + [0,1..,2m-1]*G => coset + [1,2..,2m]*G gives Tree2nd because of the interleaving property
//! FFtree with leaves given by coset + [0..m-1]*(G') => coset + [0..m-1]*(2G) => [0,2,4,..2m-4,2m-2] gives Treen
//! FFtree with leaves given by (coset+G) + [0..m-1]*(G') => (coset+G) + [0..m-1]*(2G) => [1, 3, 4, 2m-3, 2m-1] gives Treend
//! Therefore to generated tree over domain D', we shift coset by G which is the `base_generator`,
//! we use this shifted coset to shift leaves [0..n-1] G', where G' is the generator over the sub-domain

use crate::curve::Fr;
use ark_ff::AdditiveGroup;
use ark_ff::Field;
use ark_ff::Zero;
use ark_poly::DenseUVPolynomial;
use ark_poly::Polynomial;
use ark_poly::univariate::DensePolynomial;
use ark_std::str::FromStr;
use ecfft::FFTree;
use ecfft::ec::Point;
use ecfft::ec::ShortWeierstrassCurve;
use ecfft::utils::two_adicity;
use num_bigint::BigUint;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;

// generator for the additive subgroup of size 2 * num_constraints
// Fields: `subgroup_generator` and `subgroup_order` are constant parameters,
// `base_log_n` is log_2(2 * num_constraints)
fn base_generator(
    subgroup_generator: Point<ShortWeierstrassCurve<Fr>>,
    subgroup_order: usize,
    base_log_n: usize,
) -> Point<ShortWeierstrassCurve<Fr>> {
    let mut generator = subgroup_generator;
    for _ in 0..subgroup_order - base_log_n {
        generator += generator;
    }
    generator
}

// The following function: build_ec_fftrees is copied from ecfft library.
// But includes some changes to use parallel processing while computing leaves
/// Build EC FFTree for efficient polynomial operations with curve parameters and instance specific values
/// Fields:
///     # subgroup_generator, coset_offset, subgroup_order: curve constants
///     # base_log_n: 1 << num_constraints+1
///     # domain_len: size of domain for polynomial operation
///     # shift_by_one: true for domain over D', false for domain D. shift_by_one swaps odd and even indexed leaves of a tree
///     # minimal_tree: true if FFTree<Fr> instance should only include fields necessary for proof generation. Useful for memory efficiency.
fn build_ec_fftrees(
    subgroup_generator: Point<ShortWeierstrassCurve<Fr>>,
    subgroup_order: usize,
    coset_offset: Point<ShortWeierstrassCurve<Fr>>,
    domain_len: usize,
    shift_by_one: bool,
    base_log_n: usize,
    minimal_tree: bool,
) -> Option<FFTree<Fr>> {
    assert_ne!(coset_offset, subgroup_generator);
    assert_eq!(coset_offset.curve, subgroup_generator.curve);
    assert!(domain_len.is_power_of_two());
    assert!(subgroup_order.is_power_of_two());
    let subgroup_two_addicity = subgroup_order.ilog2();
    let log_n = domain_len.ilog2();
    assert!(log_n < 32);

    // FFTree size is too large for our generator
    if log_n > subgroup_two_addicity {
        return None;
    }

    // G' <- get a generator for a subgroup with order `log_n`
    let mut generator = subgroup_generator;
    for _ in 0..subgroup_two_addicity - log_n {
        generator += generator;
    }

    // `base_generator` or G <- get a generator for a subgroup with order `base_log_n`
    let base_generator = base_generator(
        subgroup_generator,
        subgroup_two_addicity as usize,
        base_log_n,
    );

    // find our rational maps; directly copied from ecfft lib
    let mut rational_maps = Vec::new();
    let mut g = generator;
    for _ in 0..log_n {
        let isogeny = g
            .curve
            .unwrap()
            .two_isogenies()
            .into_iter()
            .find_map(|isogeny| {
                let g_prime = isogeny.map(&g);
                if two_adicity(g)? == two_adicity(g_prime)? + 1 {
                    g = g_prime;
                    Some(isogeny)
                } else {
                    None
                }
            })
            .expect("cannot find a suitable isogeny");
        rational_maps.push(isogeny.r)
    }

    // for shifted domain i.e. D', add base_generator G to coset
    let coset_offset = if shift_by_one {
        coset_offset + base_generator
    } else {
        coset_offset
    };

    // generate the FFTree leaf nodes
    let mut leaves = vec![Fr::zero(); domain_len];
    leaves.par_iter_mut().enumerate().for_each(|(i, leaf)| {
        let point = coset_offset + generator * BigUint::from(i); // [C + G] + [i] G'
        *leaf = point.x;
    });

    let dat_tree = if minimal_tree {
        FFTree::new_small(leaves, rational_maps) // small tree has only fields necessary for proof generation
    } else {
        FFTree::new(leaves, rational_maps)
    };
    Some(dat_tree)
}

/// Get elements from D and D' given FFTree that includes D U D' as domain
// tree2n::leaves include D and D' such that they interleave
// e.g. D[0] is the first leaf, D'[0] is the second leaf, D[1] is the third leaf,.. ,
// D[last] is the second last leaf, D'[last] is the last leaf.
// We need this interleaving property in `tree2n` for a call to extend()
// A call to `tree2n::extend()` moves 'n' evaluations from even indexed leaves to
// 'n' evaluations over odd indexed leaves i.e.  D -> D'
pub(crate) fn get_both_domains(tree2n: &FFTree<Fr>) -> [Vec<Fr>; 2] {
    let tree2n_leaves = tree2n.f.leaves();
    let num_constraints = tree2n_leaves.len() / 2;
    let mut treen_leaves: Vec<Fr> = Vec::with_capacity(num_constraints);
    let mut treend_leaves: Vec<Fr> = Vec::with_capacity(num_constraints);
    for i in (0..num_constraints * 2).step_by(2) {
        treen_leaves.push(tree2n_leaves[i]);
        treend_leaves.push(tree2n_leaves[i + 1]);
    }
    [treen_leaves, treend_leaves]
}

/// Build Sect233k1 EC FFTree for efficient polynomial operations with sect233k1 curve parameters and instance specific values
/// Fields:
///     # base_log_n: 1 << num_constraints+1
///     # domain_len: size of domain for polynomial operation
///     # shift_by_one: true for domain over D', false for domain D. shift_by_one swaps odd and even indexed leaves of a tree
///     # minimal_tree: true if FFTree<Fr> instance should only include fields necessary for proof generation. Useful for memory efficiency.
pub(crate) fn build_sect_ecfft_tree(
    domain_len: usize,
    shift_by_one: bool,
    base_log_n: usize,
    minimal_tree: bool,
) -> Option<FFTree<Fr>> {
    // ecfft's find_curve.rs gives curve params for the necessary size of subgroup
    // for our case 28 is sufficient because we have number of constraints = ~2^23 < 2^subgroup_adic
    let subgroup_adic = 28;
    // (a, b) params in "Good Curve" form is returned by find_curve.rs
    // ecfft_curve_params_utils::get_short_weierstrass_params_from_good_curve_params converts it into SW form
    // which we hardcode below
    let curve: ShortWeierstrassCurve<Fr> = ShortWeierstrassCurve::new(
        Fr::from_str("2125753088427212854352924174339172498722499297750753614229533284661082")
            .unwrap(),
        Fr::from_str("3303427382072851929105738691313541325219445842218525662544269869787589")
            .unwrap(),
    );
    let subgroup_generator = Point::new(
        Fr::from_str("1969398527398874941115360315313056361667745675958024267654083765592400")
            .unwrap(),
        Fr::from_str("917696706299601920847965073366118878832337776859300472447868491055982")
            .unwrap(),
        curve,
    );
    // suitable coset offset is sampled using ecfft_curve_params_utils::find_coset_for_sw
    let coset_offset = Point::new(
        Fr::from_str("1557215852494830750811239888869886110709986867282698163663807961412586")
            .unwrap(),
        Fr::from_str("2302954593454110051167704558708330032236229062988890422530712548754008")
            .unwrap(),
        curve,
    );
    build_ec_fftrees(
        subgroup_generator,
        1 << subgroup_adic,
        coset_offset,
        domain_len,
        shift_by_one,
        base_log_n,
        minimal_tree,
    )
}

pub(crate) fn compute_vanishing_polynomial(
    fftree_2n: &FFTree<Fr>, // FFTree (size 2*N required for vanishing polynomial)
) -> Result<DensePolynomial<Fr>, String> {
    let n = fftree_2n.f.leaves().len() / 2;
    assert!(n > 1 && n.is_power_of_two());

    let domain_points_s = fftree_2n.subtree_with_size(n).f.leaves(); // Domain S, size N

    // Get evaluations of Z_S(X) on the leaves of fftree_2n (a 2N-point domain).
    // Z_S(X) is the vanishing polynomial for domain_points_s (N roots).
    // fftree_2n.vanish(domain_points_s) computes Z_S(X) and evaluates it on fftree_2n's leaves.
    println!("Computing evaluations of Z_S(X) using fftree_2n.vanish()...");
    let z_s_evals_on_s2n = fftree_2n.vanish(domain_points_s);
    if z_s_evals_on_s2n.len() != 2 * n && n > 0 {
        return Err(format!(
            "fftree_2n.vanish() returned {} evals, expected {}",
            z_s_evals_on_s2n.len(),
            2 * n
        ));
    }

    // Get coefficients of Z_S(X) using fftree_2n.exit().
    // Z_S(X) has degree N, so N+1 polynomial coefficients, so N+1 evaluations, so we need 2N tree
    // trees can only have power of two leaf-counts, so we require N and 2N tree, can't have N and N+1 tree
    println!("Computing coefficients of Z_S(X) using fftree_2n.exit()...");
    let mut z_s_coeffs_vec_from_exit = fftree_2n.exit(&z_s_evals_on_s2n);

    // Z_S(X) is monic of degree N. It has N+1 coefficients.
    // The `exit` function returns 2N coefficients; the higher ones should be zero.
    if z_s_coeffs_vec_from_exit.len() != 2 * n && n > 0 {
        return Err(format!(
            "fftree_2n.exit() returned {} coeffs, expected {}",
            z_s_coeffs_vec_from_exit.len(),
            2 * n
        ));
    }
    z_s_coeffs_vec_from_exit.truncate(n + 1); // Keep only the N+1 coefficients for degree N poly

    let vanishing_poly = DensePolynomial::from_coefficients_vec(z_s_coeffs_vec_from_exit);

    Ok(vanishing_poly)
}

pub(crate) fn compute_barycentric_weights(
    fftree_n: &FFTree<Fr>,
    vanishing_poly: &DensePolynomial<Fr>,
) -> Result<Vec<Fr>, String> {
    let n = fftree_n.f.leaves().len();
    // d. Compute coefficients of the derivative Z'_S(X)
    println!("  Step d: Computing derivative Z'_S(X)...");
    let z_s_prime_coeffs = derivative(vanishing_poly);
    if z_s_prime_coeffs.degree() != n - 1 {
        return Err(format!(
            "Derivative Z'_S(X) has degree {}, expected {}.",
            z_s_prime_coeffs.degree(),
            n - 1
        ));
    }

    // e. Evaluate Z'_S(X) at all points s_i in S using fftree_n.enter()
    //    Z'_S(X) has degree N-1, so N coefficients (unless N=1, then degree 0, 1 coeff).
    println!("  Step e: Evaluating Z'_S(s_i) using fftree_n.enter()...");
    let mut coeffs_for_enter = z_s_prime_coeffs.coeffs().to_vec();
    while coeffs_for_enter.len() < n {
        // Pad to size N if degree is less than N-1
        coeffs_for_enter.push(Fr::zero());
    }
    if coeffs_for_enter.len() > n && n > 0 {
        // Should not happen if padding is correct
        return Err(format!(
            "Coefficient vector for Z'_S(X) has length {}, expected at most {}.",
            coeffs_for_enter.len(),
            n
        ));
    }

    let mut z_s_prime_evals_on_s = fftree_n.enter(&coeffs_for_enter);
    if z_s_prime_evals_on_s.len() != n {
        return Err("Evaluation of Z_S_prime returned incorrect number of points.".to_string());
    }

    for (i, z_s_prime_evals_on_s_i) in z_s_prime_evals_on_s.iter().enumerate().take(n) {
        if z_s_prime_evals_on_s_i.is_zero() {
            // This implies a repeated root in S if domain_points_s were distinct,
            // or an issue with derivative calculation/evaluation.
            return Err(format!(
                "Derivative of vanishing polynomial Z'_S(s_{i}) is zero; implies repeated roots or error.",
            ));
        }
    }

    ark_ff::batch_inversion(&mut z_s_prime_evals_on_s);

    Ok(z_s_prime_evals_on_s)
}

/// Evaluates all Lagrange coefficients L_{i,S}(tau) for a domain S (leaves of fftree_n)
/// at a given point tau. Uses fftree.vanish() and fftree.exit().
/// Complexity: O(N log^2 N), potentially with larger constants due to 2N-sized tree ops.
pub(crate) fn compute_lagrange_basis_at_tau(
    fftree_n: &FFTree<Fr>, // FFTree (size 2*N required for vanishing polynomial)
    z_s_coeffs: &DensePolynomial<Fr>,
    tau: Fr,
    inv_z_s_prime_evals_on_s: &[Fr],
) -> Result<Vec<Fr>, String> {
    let n = fftree_n.f.leaves().len();
    assert!(n > 1 && n.is_power_of_two());

    let domain_points_s = fftree_n.f.leaves(); // Domain S, size N
    assert!(!domain_points_s.contains(&tau));

    if z_s_coeffs.degree() != n && n > 0 {
        // Degree should be exactly N
        return Err(format!(
            "Vanishing polynomial Z_S(X) has degree {}, expected {}. Check for distinct domain points.",
            z_s_coeffs.degree(),
            n
        ));
    }

    // c. Evaluate Z_S(tau)
    println!("Evaluating Z_S(tau)...");
    let z_s_at_tau = z_s_coeffs.evaluate(&tau);
    if z_s_at_tau.is_zero() {
        return Err(
            "Z_S(tau) is zero, but tau was not found in domain_points. This implies an issue."
                .to_string(),
        );
    }

    if inv_z_s_prime_evals_on_s.len() != n {
        return Err("Evaluation of Z_S_prime returned incorrect number of points.".to_string());
    }

    // f. Compute L_{i,S}(tau) = Z_S(tau) / ( (tau - s_i) * Z'_S(s_i) )
    println!("Computing final Lagrange coefficients...");
    let mut tau_minus_si: Vec<Fr> = Vec::with_capacity(n);
    for domain_points_s_i in domain_points_s.iter().take(n) {
        let tau_minus_s_i = tau - domain_points_s_i;
        tau_minus_si.push(tau_minus_s_i);
    }
    ark_ff::batch_inversion(&mut tau_minus_si);

    let mut lagrange_coeffs: Vec<Fr> = Vec::with_capacity(n);
    for i in 0..n {
        lagrange_coeffs.push(z_s_at_tau * tau_minus_si[i] * inv_z_s_prime_evals_on_s[i]);
    }

    Ok(lagrange_coeffs)
}

// copied from ecfft lib
// uses standard derivative formula
fn derivative<F: Field>(p: &DensePolynomial<F>) -> DensePolynomial<F> {
    if p.is_zero() || p.degree() == 1 {
        DensePolynomial::zero()
    } else {
        let mut coeffs = Vec::with_capacity(p.degree());
        for (i, coeff) in p.coeffs.iter().enumerate().skip(1) {
            coeffs.push(*coeff * F::from(i as u64));
        }
        DensePolynomial::from_coefficients_vec(coeffs)
    }
}

/// Evaluate vanishing polynomial over some domain
pub(crate) fn evaluate_vanishing_poly_at_domain(z_poly: &[Fr], treen: &FFTree<Fr>) -> Vec<Fr> {
    let treen_leaves = treen.f.leaves();
    // O(Nlog^n) cost to evaluate at leaves
    // N coefficients (N-1 degree polynomial) to begin with
    let mut tree_ev = treen.enter(&z_poly[1..]);
    // afterwards A(alpha) = [An-1(alpha)].alpha + a0
    // at linear cost
    for i in 0..treen_leaves.len() {
        let leaf = treen_leaves[i];
        tree_ev[i] = tree_ev[i] * leaf + z_poly[0];
    }
    tree_ev
}


#[allow(clippy::too_many_arguments)]
// we additionally use vanishing polynomial evaluations `z_poly_dom2` and `z_poly2_dom`
// for lesser compute cost
pub(crate) fn compute_lagrange_basis_at_tau_over_unified_domain(
    tau: Fr,
    num_constraints: usize,
    l_tau: &[Fr],
    l_tau2: &[Fr],
    z_poly: &[Fr],
    z_poly2: &[Fr],
    z_poly_dom2: &[Fr],
    z_poly2_dom: &[Fr],
) -> Vec<Fr> {
    let z_poly_tau = (DensePolynomial {
        coeffs: z_poly.to_vec(),
    })
    .evaluate(&tau);
    let z_poly2_tau = (DensePolynomial {
        coeffs: z_poly2.to_vec(),
    })
    .evaluate(&tau);

    let mut lagrange_out: Vec<Fr> = vec![Fr::ZERO; 2 * num_constraints];

    for i in 0..num_constraints {
        lagrange_out[2 * i] = l_tau[i] * z_poly2_tau * z_poly2_dom[i];
        lagrange_out[2 * i + 1] = l_tau2[i] * z_poly_tau * z_poly_dom2[i];
    }
    lagrange_out
}

// Using domain points and evaluations of a polynomial at those points "p_evals"
// obtain p(alpha) at O(N) cost -> possible using precomputed barycentric weights
// else FFT-like direct approach is quasi-linear (costlier)
pub(crate) fn evaluate_poly_at_alpha_using_barycentric_weights(
    domain_points: &[Fr],
    bar_weights: &[Fr],
    zpoly_coeffs: &DensePolynomial<Fr>,
    p_evals: &[Fr],
    alpha: Fr,
) -> Fr {
    let n = domain_points.len();
    assert_eq!(
        p_evals.len(),
        n,
        "Number of evaluations does not match the precomputation domain size."
    );

    // Handle special case where alpha is in the domain: done outside of this function

    // General case: P(alpha) = Z_S(alpha) * sum_{i=0 to N-1} [ y_i * w_i / (alpha - s_i) ]
    // where y_i are p_evals and w_i are barycentric_weights.

    // 1. Evaluate Z_S(alpha). O(N)
    let z_s_at_alpha = zpoly_coeffs.evaluate(&alpha);

    // 2. Compute the terms 1 / (alpha - s_i). O(N) + batch inversion
    let mut alpha_minus_si_inv = Vec::with_capacity(n);
    for s_i in domain_points {
        alpha_minus_si_inv.push(alpha - s_i);
    }
    ark_ff::batch_inversion(&mut alpha_minus_si_inv);

    // 3. Compute the sum. O(N)
    let mut inner_sum = Fr::zero();
    for i in 0..n {
        let term = p_evals[i] * bar_weights[i] * alpha_minus_si_inv[i];
        inner_sum += term;
    }
    z_s_at_alpha * inner_sum
}

#[cfg(test)]
mod ecfft_curve_params_utils {
    use crate::curve::Fr;
    use ark_ff::Zero;
    use ark_ff::{Field, UniformRand};
    use ark_std::rand::thread_rng;
    use std::str::FromStr;

    fn get_short_weierstrass_params_from_good_curve_params(a: &Fr, b: &Fr) -> (Fr, Fr, Fr) {
        // pre-compute constants
        let inv3 = Fr::from(3).inverse().unwrap(); // 3⁻¹  mod r
        let inv9 = inv3 * inv3; // 1/9
        let inv27 = inv9 * inv3; // 1/27   ( = 3⁻³ )

        // A = b − a²/3
        let a_sq_over3 = (a.square()) * inv3;
        let big_a = b - &a_sq_over3;

        // B = 2a³/27 − a b /3
        let two = Fr::from(2);
        let a_cu = a.square() * a; // a³
        let two_a3_over27 = two * a_cu * inv27; // 2a³/27
        let ab_over3 = (a * b) * inv3; // a b /3
        let big_b = two_a3_over27 - ab_over3;

        // shift for points:  a/3
        let x_shift = a * &inv3;

        (big_a, big_b, x_shift) //  y stays the same
    }

    fn find_coset_for_sw(param_a_str: &str, param_b_str: &str) -> (Fr, Fr) {
        let a = Fr::from_str(param_a_str).unwrap();
        let b = Fr::from_str(param_b_str).unwrap();

        // 2. Search for a random point ≠ (0,0) to act as coset offset

        let mut rng = thread_rng();
        let mut iterations = 0u64;

        #[allow(unused_assignments)]
        let mut found: Option<(Fr, Fr)> = None;

        loop {
            iterations += 1;
            let x = Fr::rand(&mut rng);

            // y^2 = x^3 + A·x + B
            let rhs = x.square() * x + a * x + b;

            if let Some(y) = rhs.sqrt() {
                if x.is_zero() && y.is_zero() {
                    continue;
                }
                found = Some((x, y));
                break;
            }

            if iterations > 1_000_000 {
                panic!("searched >1 M candidates without finding a point – unexpected");
            }
        }

        let (px, py) = found.expect("no point found");

        // 3. Sanity check:  does the point really satisfy the SW equation?

        let lhs = py.square();
        let rhs = px.square() * px + a * px + b;
        assert_eq!(lhs, rhs, "point is NOT on the curve!");
        (px, py)
    }

    mod test {
        use crate::{
            curve::Fr,
            ec_fft::ecfft_curve_params_utils::{
                find_coset_for_sw, get_short_weierstrass_params_from_good_curve_params,
            },
        };
        use num_bigint::BigUint;
        use std::str::FromStr;

        #[test]
        fn test_sect_get_short_weierstrass_params_from_good_curve_params() {
            // 2^28 subgroup order
            let a = Fr::from(
                BigUint::from_str(
                    "1612258558012370015697612802162317394937170966914298907792172265935528",
                )
                .unwrap(),
            );
            let b = Fr::from(
                BigUint::from_str(
                    "1649539414812869195553934139907948967057031997387519615898515827272474",
                )
                .unwrap(),
            );
            let (a_sw, b_sw, x_shift) = get_short_weierstrass_params_from_good_curve_params(&a, &b);

            let px = Fr::from(
                BigUint::from_str(
                    "1431979008061418269216156047925617230022022020319924631723359676947224",
                )
                .unwrap(),
            );
            let py = Fr::from(
                BigUint::from_str(
                    "917696706299601920847965073366118878832337776859300472447868491055982",
                )
                .unwrap(),
            );

            let px_sw = px + x_shift;
            let py_sw = py;

            println!("a {}", a_sw);
            println!("b {}", b_sw);
            println!("px {}", px_sw);
            println!("py {}", py_sw);
        }

        #[test]
        fn test_find_coset_for_sw() {
            let param_a_str =
                "2125753088427212854352924174339172498722499297750753614229533284661082";
            let param_b_str =
                "3303427382072851929105738691313541325219445842218525662544269869787589";

            let coset = find_coset_for_sw(param_a_str, param_b_str);
            println!("coset {:?}", coset);
        }
    }
}

#[cfg(test)]
mod ecfft_properties {
    use super::build_sect_ecfft_tree;

    #[test]
    fn test_subtree() {
        let base_log_n = 7;
        let num_constraints = 1 << 6;
        let tree2n = build_sect_ecfft_tree(num_constraints * 2, false, base_log_n, false).unwrap();
        let tree2n_subtree = tree2n.subtree_with_size(num_constraints);
        let treen = build_sect_ecfft_tree(num_constraints, false, base_log_n, false).unwrap();
        assert_eq!(tree2n_subtree.f.leaves(), treen.f.leaves()); // assertion passes

        let tree2n = build_sect_ecfft_tree(num_constraints * 2, true, base_log_n, false).unwrap();
        let tree2n_subtree = tree2n.subtree_with_size(num_constraints);
        let treen = build_sect_ecfft_tree(num_constraints, true, base_log_n, false).unwrap();
        assert_eq!(tree2n_subtree.f.leaves(), treen.f.leaves()); // assertion fails
    }
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use super::{build_sect_ecfft_tree, compute_lagrange_basis_at_tau};
    use crate::artifacts::TREE_2N;
    use crate::curve::Fr;
    use crate::ec_fft::{
        compute_barycentric_weights, compute_lagrange_basis_at_tau_over_unified_domain, evaluate_vanishing_poly_at_domain, compute_vanishing_polynomial
    };
    use crate::tree_io::{read_fftree_from_file, write_fftree_to_file};

    use ark_ff::Zero;
    use ark_ff::{Field, One, UniformRand};
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::{DenseUVPolynomial, Polynomial};
    use ark_std::rand::{SeedableRng, thread_rng};
    use ecfft::FFTree;
    use rand_chacha::ChaCha20Rng;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    #[test]
    fn test_evaluate_lagrange_coeffs_using_precompute() {
        let n_log: u32 = 4; // Keep N small for faster FFTree builds in this example
        let n = 1 << n_log;

        println!("Building FFTree (fftree_n) of size {} for EcfftFp...", n);
        let fftree_2n = build_sect_ecfft_tree(2 * n, false, n_log as usize + 1, false)
            .expect("Failed to build fftree_n");
        println!("FFTree (fftree_n) built successfully.");
        let fftree_n = fftree_2n.subtree_with_size(n);

        let domain_s = fftree_n.f.leaves().to_vec();

        let mut tau = Fr::from(123u64); // A different tau
        if domain_s.contains(&tau) {
            println!(
                "Tau ({}) is in the domain. Modifying tau to be outside.",
                tau
            );
            tau += Fr::from(domain_s.len() as u64 + 1); // Try to jump further out
            println!("New tau: {}", tau);
        }
        if domain_s.contains(&tau) {
            // Final check
            tau += Fr::one(); // One last attempt
        }
        if n > 0 {
            // Only assert if domain is non-empty
            assert!(
                !domain_s.contains(&tau),
                "Tau should not be in the domain for this test path"
            );
        }

        println!(
            "\nEvaluating Lagrange coefficients at tau = {} (tau is NOT in domain S):",
            tau
        );

        let vanishing_poly = compute_vanishing_polynomial(&fftree_2n).unwrap();
        let barycentric_weight = compute_barycentric_weights(fftree_n, &vanishing_poly).unwrap();

        match compute_lagrange_basis_at_tau(
            fftree_n,
            &vanishing_poly,
            tau,
            &barycentric_weight,
        ) {
            Ok(lagrange_basis_at_tau) => {
                if n > 0 {
                    // Only verify if domain and coeffs are non-empty
                    let mut p_evals_on_s = Vec::with_capacity(n);
                    for s_i in &domain_s {
                        p_evals_on_s.push(*s_i + Fr::from(7u32)); // P(X) = X + 7
                    }

                    let mut p_at_tau_reconstructed = Fr::zero();
                    for (i, p_evals_on_s_i) in p_evals_on_s.iter().enumerate().take(n) {
                        p_at_tau_reconstructed += lagrange_basis_at_tau[i] * p_evals_on_s_i;
                    }

                    let p_at_tau_direct = tau + Fr::from(7u32);

                    println!("  P(X) = X + 7");
                    println!("  P(tau) directly: {}", p_at_tau_direct);
                    println!(
                        "  P(tau) reconstructed with Lagrange coeffs: {}",
                        p_at_tau_reconstructed
                    );

                    let difference = if p_at_tau_direct > p_at_tau_reconstructed {
                        // Naive abs for Fp
                        p_at_tau_direct - p_at_tau_reconstructed
                    } else {
                        p_at_tau_reconstructed - p_at_tau_direct
                    };

                    if difference.is_zero() {
                        println!("  Verification successful!");
                    } else {
                        eprintln!("  Verification FAILED! Difference: {}", difference);
                    }
                } else {
                    println!("  Domain is empty, no verification performed.");
                }
            }
            Err(e) => eprintln!("Error in general case: {}", e),
        }
    }

    #[test]
    fn test_compare_with_bruteforce() {
        /// Brute-force barycentric formula:
        ///   L_i(τ) = ∏_{j≠i} (τ − s_j) / ∏_{j≠i} (s_i − s_j)
        fn lagrange_coeffs_reference<F: Field>(domain: &[F], tau: F) -> Vec<F> {
            let n = domain.len();
            // Pre-compute all denominators D_i = ∏_{j≠i}(s_i - s_j)
            let mut denom = vec![F::one(); n];
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        denom[i] *= domain[i] - domain[j];
                    }
                }
            }
            ark_ff::batch_inversion(&mut denom); // 1 inversion + O(n) muls

            // Common numerator factor  Z_S(τ) = ∏_{j}(τ − s_j)
            let mut z_tau = F::one();
            for s in domain {
                z_tau *= tau - s;
            }

            // Final L_i
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(z_tau * (tau - domain[i]).inverse().unwrap() * denom[i]);
            }
            out
        }

        // --- choose a small power-of-two domain ---------------------------
        let n_log = 4u32; //  N = 16   (fits in a few ms)
        let n = 1 << n_log;

        let fftree2n = build_sect_ecfft_tree(2 * n, false, n_log as usize + 1, false)
            .expect("cannot build FFTree");
        let fftree = fftree2n.subtree_with_size(n);
        let domain_s = fftree.f.leaves(); // &[F; N]

        // --- pick τ  (random, but retry until τ ∉ S) ----------------------
        let mut rng = thread_rng();
        let mut tau = Fr::rand(&mut rng);
        while domain_s.contains(&tau) {
            tau = Fr::rand(&mut rng);
        }

        // --- fast ECFFT implementation ------------------------------------
        let vanishing_poly = compute_vanishing_polynomial(&fftree2n).unwrap();
        let barycentric_weight = compute_barycentric_weights(fftree, &vanishing_poly).unwrap();

        let lagrange_basis_at_tau = compute_lagrange_basis_at_tau(
            fftree,
            &vanishing_poly,
            tau,
            &barycentric_weight,
        )
        .expect("ECFFT routine failed");

        // --- slow reference -----------------------------------------------
        let slow = lagrange_coeffs_reference(domain_s, tau);

        assert_eq!(lagrange_basis_at_tau.len(), slow.len());
        for (i, (f, s)) in lagrange_basis_at_tau.iter().zip(slow.iter()).enumerate() {
            assert_eq!(f, s, "Mismatch at index {i}:  fast = {f},  slow = {s}");
        }
    }

    #[test]
    fn test_vanishing_poly() {
        let n_log = 4u32; //  N = 16   (fits in a few ms)
        let n = 1 << n_log;

        let fftree_2n = build_sect_ecfft_tree(n * 2, false, n_log as usize + 1, false)
            .expect("cannot build FFTree");
        let fftree_n = build_sect_ecfft_tree(n, false, n_log as usize + 1, false)
            .expect("cannot build FFTree");

        let vanishing_poly = compute_vanishing_polynomial(&fftree_2n).unwrap();

        let domain_s = fftree_n.f.leaves(); // &[F; N]

        /// Compute vanishing polynomial z(X) = product(X - r_i) for the domain.
        fn ark_compute_vanishing_polynomial(dom: &[Fr]) -> DensePolynomial<Fr> {
            fn dense_poly_mul_parallel<F: Field + Send + Sync>(a: &[F], b: &[F]) -> Vec<F> {
                let result_len = a.len() + b.len() - 1;

                // Each thread computes a local result vector of the same size
                let partial_results: Vec<Vec<F>> = a
                    .par_iter()
                    .enumerate()
                    .map(|(i, coeff_a)| {
                        let mut local = vec![F::ZERO; result_len];
                        for (j, coeff_b) in b.iter().enumerate() {
                            local[i + j] = *coeff_a * *coeff_b;
                        }
                        local
                    })
                    .collect();

                // Reduce all partial results into the final result vector
                let mut result = vec![F::ZERO; result_len];
                for local in partial_results {
                    for (i, coeff) in local.into_iter().enumerate() {
                        result[i] += coeff;
                    }
                }

                result
            }

            let mut poly_z = DensePolynomial::from_coefficients_vec(vec![Fr::one()]);
            for &r in dom {
                let linear_factor = DensePolynomial::from_coefficients_vec(vec![-r, Fr::one()]); // (X - r)
                let tmp = dense_poly_mul_parallel(&poly_z.coeffs, &linear_factor.coeffs);
                poly_z = DensePolynomial { coeffs: tmp };
            }

            poly_z
        }

        let ark_vpoly = ark_compute_vanishing_polynomial(domain_s);

        assert_eq!(ark_vpoly, vanishing_poly);

        for d in domain_s {
            let evals = vanishing_poly.evaluate(d);
            assert!(evals.is_zero());
        }
    }

    #[test]
    fn test_interpolate_and_extend_match() {
        let n_log = 4u32; //  N = 16   (fits in a few ms)
        let domain_len = 1 << n_log;

        let fftree_2n = build_sect_ecfft_tree(2 * domain_len, false, n_log as usize + 1, false)
            .expect("cannot build FFTree");
        let fftree_2nd = build_sect_ecfft_tree(2 * domain_len, true, n_log as usize + 1, false)
            .expect("cannot build FFTree");
        let fftree_nd = fftree_2nd.subtree_with_size(domain_len);

        let mut rng = thread_rng();
        let evals: Vec<Fr> = (0..domain_len).map(|_| Fr::rand(&mut rng)).collect();
        let domaind = fftree_nd.f.leaves();

        let poly = fftree_2n.exit(&evals);
        let ark_poly = DensePolynomial { coeffs: poly };
        let mut ark_evalds = vec![];
        for leaf in domaind {
            let evald = ark_poly.evaluate(leaf);
            ark_evalds.push(evald);
        }

        let ec_evalds = fftree_2n.extend(&evals, ecfft::Moiety::S1);
        assert_eq!(ec_evalds, ark_evalds);
    }

    #[test]
    fn test_compare_with_bruteforce2() {
        /// Brute-force barycentric formula:
        ///   L_i(τ) = ∏_{j≠i} (τ − s_j) / ∏_{j≠i} (s_i − s_j)
        fn lagrange_coeffs_reference<F: Field>(domain: &[F], tau: F) -> Vec<F> {
            let n = domain.len();
            // Pre-compute all denominators D_i = ∏_{j≠i}(s_i - s_j)
            let mut denom = vec![F::one(); n];
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        denom[i] *= domain[i] - domain[j];
                    }
                }
            }
            ark_ff::batch_inversion(&mut denom); // 1 inversion + O(n) muls

            // Common numerator factor  Z_S(τ) = ∏_{j}(τ − s_j)
            let mut z_tau = F::one();
            for s in domain {
                z_tau *= tau - s;
            }

            // Final L_i
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(z_tau * (tau - domain[i]).inverse().unwrap() * denom[i]);
            }
            out
        }

        // --- choose a small power-of-two domain ---------------------------
        let n_log = 4u32; //  N = 16   (fits in a few ms)
        let n = 1 << n_log;

        let fftreen = build_sect_ecfft_tree(2 * n, false, n_log as usize + 1, false)
            .expect("cannot build FFTree");
        let domain_s = fftreen.f.leaves(); // &[F; N]
        let (e0, domain_s): (Vec<Fr>, Vec<Fr>) = domain_s.chunks(2).map(|e| (e[0], e[1])).unzip();
        assert_eq!(e0, fftreen.subtree_with_size(n).f.leaves());

        // --- pick τ  (random, but retry until τ ∉ S) ----------------------
        let mut rng = thread_rng();
        let mut tau = Fr::rand(&mut rng);
        while domain_s.contains(&tau) {
            tau = Fr::rand(&mut rng);
        }

        // --- fast ECFFT implementation ------------------------------------
        let fftree2n = build_sect_ecfft_tree(2 * n, true, n_log as usize + 1, false)
            .expect("cannot build FFTree");
        // let fftree2n_sub = fftree2n.subtree_with_size(n);
        let domain_sd = fftree2n.subtree_with_size(n).f.leaves();
        assert_eq!(domain_s, domain_sd);

        let vpoly = compute_vanishing_polynomial(&fftree2n).unwrap();
        let fftreen = fftree2n.subtree_with_size(n);
        let bar_wts = compute_barycentric_weights(fftreen, &vpoly).unwrap();
        let fast = compute_lagrange_basis_at_tau(fftreen, &vpoly, tau, &bar_wts)
            .expect("ECFFT routine failed");
        // // --- slow reference -----------------------------------------------
        let slow = lagrange_coeffs_reference(domain_sd, tau);

        assert_eq!(fast.len(), slow.len());
        for (i, (f, s)) in fast.iter().zip(slow.iter()).enumerate() {
            assert_eq!(f, s, "Mismatch at index {i}:  fast = {f},  slow = {s}");
        }
    }

    #[test]
    fn test_combine_smaller_lagranges() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let tau = Fr::rand(&mut rng);

        let n_log = 6;
        let num_constraints = 1 << n_log;

        let tree2n =
            build_sect_ecfft_tree(num_constraints * 2, false, n_log as usize + 1, false).unwrap();
        let tree2nd =
            build_sect_ecfft_tree(num_constraints * 2, true, n_log as usize + 1, false).unwrap();
        let tree4n =
            build_sect_ecfft_tree(num_constraints * 2 * 2, false, n_log as usize + 1, false)
                .unwrap();

        let z_poly = compute_vanishing_polynomial(&tree2n).unwrap();
        let treen = tree2n.subtree_with_size(num_constraints);
        let bar_wts = compute_barycentric_weights(treen, &z_poly).unwrap();
        let l_tau =
            compute_lagrange_basis_at_tau(treen, &z_poly, tau, &bar_wts).unwrap();

        let z_poly2 = compute_vanishing_polynomial(&tree2nd).unwrap();
        let treend = tree2nd.subtree_with_size(num_constraints);
        let bar_wts2 = compute_barycentric_weights(treend, &z_poly2).unwrap();
        let l_tau2 =
            compute_lagrange_basis_at_tau(treend, &z_poly2, tau, &bar_wts2).unwrap();

        let z_polyl = compute_vanishing_polynomial(&tree4n).unwrap();
        let bar_wtsl = compute_barycentric_weights(&tree2n, &z_polyl).unwrap();

        let l_tau_ref =
            compute_lagrange_basis_at_tau(&tree2n, &z_polyl, tau, &bar_wtsl).unwrap();

        let mut z_poly_dom2 = evaluate_vanishing_poly_at_domain(&z_poly, treend);
        ark_ff::batch_inversion(&mut z_poly_dom2);
    
        let mut z_poly2_dom = evaluate_vanishing_poly_at_domain(&z_poly2, treen);
        ark_ff::batch_inversion(&mut z_poly2_dom);

        
        let lagrange_out = compute_lagrange_basis_at_tau_over_unified_domain(
            tau, num_constraints, &l_tau, &l_tau2, &z_poly, &z_poly2, &z_poly_dom2, &z_poly2_dom
        );

        assert_eq!(l_tau_ref, lagrange_out);
    }

    fn get_treen_nd_leaves_from_2n(tree2n: &FFTree<Fr>) -> (Vec<Fr>, Vec<Fr>) {
        let tree2n_leaves = tree2n.f.leaves();
        let two_n = tree2n_leaves.len();
        let mut treen_leaves: Vec<Fr> = Vec::with_capacity(two_n / 2);
        let mut treend_leaves: Vec<Fr> = Vec::with_capacity(two_n / 2);
        for i in (0..tree2n_leaves.len()).step_by(2) {
            treen_leaves.push(tree2n_leaves[i]);
            treend_leaves.push(tree2n_leaves[i + 1]);
        }
        (treen_leaves, treend_leaves)
    }

    #[test]
    fn test_union_of_sub_tree_leaves() {
        let n_log: usize = 3;
        let num_constraints = 1 << n_log;
        let tree2n = build_sect_ecfft_tree(num_constraints * 2, false, n_log + 1, false).unwrap();
        let tree2nd = build_sect_ecfft_tree(num_constraints * 2, true, n_log + 1, false).unwrap();
        let treen_leaves = tree2n.subtree_with_size(num_constraints).f.leaves();
        let treend_leaves = tree2nd.subtree_with_size(num_constraints).f.leaves();

        let (calc_treen_leaves, calc_treend_leaves) = get_treen_nd_leaves_from_2n(&tree2n);
        assert_eq!(calc_treen_leaves, treen_leaves.to_vec());
        assert_eq!(calc_treend_leaves, treend_leaves.to_vec());
    }

    #[test]
    fn test_evaluate_lagrange_coeffs_using_precompute2() {
        let n_log: usize = 6;
        let domain_len = 1 << n_log;
        let fftree_2n = build_sect_ecfft_tree(domain_len, false, n_log + 1, false).unwrap();
        let mut rng = thread_rng();
        let tau = Fr::rand(&mut rng);

        let z_poly = compute_vanishing_polynomial(&fftree_2n).unwrap();
        let fftree_n = fftree_2n.subtree_with_size(domain_len / 2);
        let z_s_prime_evals_on_s = compute_barycentric_weights(fftree_n, &z_poly).unwrap();
        let lag_coeff = compute_lagrange_basis_at_tau(
            fftree_n,
            &z_poly,
            tau,
            &z_s_prime_evals_on_s,
        )
        .unwrap();

        let lag_coeff2 = compute_lagrange_basis_at_tau(
            fftree_n,
            &z_poly,
            tau,
            &z_s_prime_evals_on_s,
        )
        .unwrap();
        assert_eq!(lag_coeff, lag_coeff2);
        let delta = Fr::rand(&mut rng);
        let lag_coeff3 = compute_lagrange_basis_at_tau(
            fftree_n,
            &z_poly,
            delta,
            &z_s_prime_evals_on_s,
        )
        .unwrap();

        let lag_coeff4 = compute_lagrange_basis_at_tau(
            fftree_n,
            &z_poly,
            delta,
            &z_s_prime_evals_on_s,
        )
        .unwrap();
        assert_eq!(lag_coeff3, lag_coeff4);
    }

    #[test]
    #[ignore]
    fn test_utils_dump_half_tree() {
        let cache_dir = Path::new("srs_sect");
        let tree2nd: FFTree<Fr> = read_fftree_from_file(cache_dir.join(TREE_2N)).unwrap();
        let leaf_count = tree2nd.f.leaves().len();
        let treend = tree2nd.subtree_with_size(leaf_count / 2).clone();
        write_fftree_to_file(&treend, cache_dir.join(TREE_2N)).unwrap();
    }
}
