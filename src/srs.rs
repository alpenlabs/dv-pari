//! Setup Phase and Proof Verification
//!
use crate::artifacts::{
    BAR_WTS, BAR_WTSD, R1CS_CONSTRAINTS_FILE, SRS_G_K_0, SRS_G_K_1, SRS_G_K_2, SRS_G_M, SRS_G_Q,
    TREE_2N, TREE_2ND, TREE_N, TREE_ND, Z_POLY, Z_POLYD, Z_VALS2_INV, Z_VALS2D_INV,
};
use crate::curve::{CurvePoint, Fr, multi_scalar_mul, point_scalar_mul_gen};
use crate::ec_fft::{
    build_sect_ecfft_tree, compute_barycentric_weights, compute_lagrange_basis_at_tau,
    compute_lagrange_basis_at_tau_over_unified_domain, compute_vanishing_polynomial,
    evaluate_vanishing_poly_at_domain,
};
use crate::gnark_r1cs::{
    R1CSInstance, Row, evaluate_monomial_basis_poly, load_sparse_r1cs_from_file,
};
use crate::io_utils::{read_fr_vec_from_file, write_fr_vec_to_file, write_point_vec_to_file};
use crate::proving::{Proof, Transcript};
use crate::tree_io::{read_fftree_from_file, write_fftree_to_file};
use anyhow::{Context, Result};
use ark_ff::{Field, One, Zero};
use ark_poly::Polynomial;
use ark_poly::univariate::DensePolynomial;
use ark_std::vec::Vec;
use ecfft::FFTree;
use ecfft::utils::BinaryTree;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::fs::File;
use std::path::Path;

/// Structured Reference String
#[derive(Clone, Debug)]
pub struct SRS {
    /// g_m ε·m_j(τ,δ)·G
    pub g_m: Vec<CurvePoint>, // num_inputs bases
    /// g_q ε·z(τ)·L_i(τ)·G
    pub g_q: Vec<CurvePoint>, // num_constraints bases
    /// g_k L_i(τ)·δ^j·G
    pub g_k: [Vec<CurvePoint>; 3], // three degree‑(m−1) bases
}

/// Trapdoor only known to the verifier
#[derive(Clone, Debug, Copy)]
pub struct Trapdoor {
    /// tau
    pub tau: Fr,
    /// delta
    pub delta: Fr,
    /// epsilon
    pub epsilon: Fr,
}

/// Pre-compute  m_j(τ,δ) = Σ_i (A_ij + δB_ij + δ^2 C_ij
pub(crate) fn accumulate_m_values(rows: &[Row], coeffs: &[Fr], l_tau: &[Fr], delta: Fr) -> Vec<Fr> {
    //assert_eq!(rows.len(), l_tau.len(), "rows / l_tau length mismatch");

    let n = rows
        .iter()
        .flat_map(|r| r.l.iter().chain(&r.r).chain(&r.o))
        .map(|t| t.wire_id as usize)
        .max()
        .unwrap_or(0)
        + 1;

    let mut m_vals = vec![Fr::zero(); n];
    let delta2 = delta.square();

    for (i, row) in rows.iter().enumerate() {
        let lt = l_tau[i];
        let sc_a = lt; // 1
        let sc_b = lt * delta; // δ
        let sc_c = lt * delta2; // δ^2

        for t in &row.l {
            m_vals[t.wire_id as usize] += coeffs[t.coeff_id as usize] * sc_a;
        }
        for t in &row.r {
            m_vals[t.wire_id as usize] += coeffs[t.coeff_id as usize] * sc_b;
        }
        for t in &row.o {
            m_vals[t.wire_id as usize] += coeffs[t.coeff_id as usize] * sc_c;
        }
    }
    m_vals
}

fn clear_fftree(tree: &mut FFTree<Fr>) {
    tree.recombine_matrices = BinaryTree::from(vec![]);
    tree.f = BinaryTree::from(vec![]);
    tree.decompose_matrices = BinaryTree::from(vec![]);

    tree.rational_maps.clear();
    tree.subtree = None;
    tree.xnn_s.clear();
    tree.xnn_s_inv.clear();
    tree.z0_s1.clear();
    tree.z1_s0.clear();
    tree.z0_inv_s1.clear();
    tree.z1_inv_s0.clear();
    tree.z0z0_rem_xnn_s.clear();
    tree.z1z1_rem_xnn_s.clear();
}

struct SRSMatrices {
    g_m: Vec<CurvePoint>,
    g_q: Vec<CurvePoint>,
    g_k: [Vec<CurvePoint>; 3],
}

/// Builds the SRS matrices required by the verifier.  All intermediate artefacts
/// are cached on disk under `cache_dir` so that subsequent runs can skip
/// expensive recomputation.
fn compute_srs_matrices(
    cache_dir: &Path,
    secrets: &Trapdoor,
    z_poly_ark: &DensePolynomial<Fr>,
    inst: &mut R1CSInstance, // mutable to free up once used
    l_tau: &[Fr],
    l_taud: &[Fr],
    l_taul: &[Fr],
) -> Result<SRSMatrices> {
    let (tau, delta) = (secrets.tau, secrets.delta);
    let z_tau = z_poly_ark.evaluate(&tau);
    let delta2 = delta.square();

    // g_m -------------------------------------------------------------------
    let g_m: Vec<CurvePoint> = {
        let m_vals = accumulate_m_values(&inst.rows, &inst.coeffs, l_tau, delta);
        inst.rows.clear();
        inst.coeffs.clear();
        m_vals
            .into_par_iter()
            .map(|val| point_scalar_mul_gen(val * secrets.epsilon))
            .collect()
    };
    write_point_vec_to_file(cache_dir.join(SRS_G_M), &g_m)?;

    // g_q -------------------------------------------------------------------
    let g_q: Vec<CurvePoint> = (0..inst.num_constraints)
        .into_par_iter()
        .map(|i| {
            let scalar = z_tau * delta2 * l_taud[i] * secrets.epsilon;
            point_scalar_mul_gen(scalar)
        })
        .collect();
    write_point_vec_to_file(cache_dir.join(SRS_G_Q), &g_q)?;

    // g_k vectors ------------------------------------------------------------
    let delta_pows = [Fr::one(), delta, delta2];
    let mut g_k_vecs: [Vec<CurvePoint>; 3] = Default::default();
    const SRS_GK: [&str; 3] = [SRS_G_K_0, SRS_G_K_1, SRS_G_K_2];
    for (j, g_k) in g_k_vecs.iter_mut().enumerate() {
        let path = cache_dir.join(SRS_GK[j]);
        let l_slice = if j < 2 { l_tau } else { l_taul };

        *g_k = l_slice
            .into_par_iter()
            .map(|l_val| point_scalar_mul_gen(*l_val * delta_pows[j]))
            .collect();
        write_point_vec_to_file(path, g_k)?;
    }

    Ok(SRSMatrices {
        g_m,
        g_q,
        g_k: g_k_vecs,
    })
}

impl SRS {
    /// Verifier runs SRS setup
    /// Fields
    ///     # trapdoor: Trapdoor secrets
    ///     # cache_dir: path verifier references to read and write data
    ///     # num_public_inputs: number of public inputs
    ///     # is_fresh_setup: Whether verifier has downloaded some precomputed data or wants to compute everything on his own
    ///     # validate_precompute: whether verifier trusts the provider or he wants to validate the data despite having the claimed shasum of the precomputed content
    pub fn verifier_runs_setup(
        trapdoor: Trapdoor,
        cache_dir: &Path,
        num_public_inputs: usize,
        is_fresh_setup: bool,
        validate_precompute: bool,
    ) -> Result<Self> {
        std::fs::create_dir_all(cache_dir) // ensure directory exists
            .with_context(|| format!("creating {}", cache_dir.display()))?;

        let mut inst = {
            let dump = load_sparse_r1cs_from_file(
                File::open(cache_dir.join(R1CS_CONSTRAINTS_FILE)).unwrap(),
            )
            .unwrap();
            R1CSInstance::from_dump(dump.clone(), num_public_inputs)
        };

        let num_constraints = inst.num_constraints;
        let n_log = num_constraints.ilog2() as usize;

        // validate trapdoor
        assert!(!trapdoor.tau.is_zero(), "tau must be non-zero");
        assert!(!trapdoor.delta.is_zero(), "delta must be non-zero");
        assert!(!trapdoor.epsilon.is_zero(), "epsilon must be non-zero");

        let load_tree = |name: &str, odd_leaves: bool, num_leaves: usize| -> Result<FFTree<Fr>> {
            let path = cache_dir.join(name);
            if path.exists() {
                println!("Reading {:?} …", path.display());
                read_fftree_from_file(&path)
            } else {
                println!("Computing {:?} …", path.display());
                let tree = build_sect_ecfft_tree(num_leaves, odd_leaves, n_log + 1, false).unwrap();
                write_fftree_to_file(&tree, &path)?;
                Ok(tree)
            }
        };

        let init_tree_and_poly_at_tau =
            |tree2nf,
             treenf,
             zpolyf,
             barwtsf,
             odd_leaf|
             -> Result<(FFTree<Fr>, Vec<Fr>, DensePolynomial<Fr>)> {
                let (treen, vanishing_poly) = if is_fresh_setup {
                    let tree = load_tree(tree2nf, odd_leaf, num_constraints * 2).unwrap();
                    let vanish_poly: DensePolynomial<Fr> = if cache_dir.join(zpolyf).exists() {
                        // read cached file if it exists from previous run
                        let zpoly_coeff = read_fr_vec_from_file(cache_dir.join(zpolyf)).unwrap();
                        DensePolynomial {
                            coeffs: zpoly_coeff,
                        }
                    } else {
                        // generate fresh
                        compute_vanishing_polynomial(&tree).unwrap()
                    };
                    let tree: FFTree<Fr> = tree.subtree_with_size(num_constraints).clone();
                    write_fr_vec_to_file(cache_dir.join(zpolyf), &vanish_poly.coeffs).unwrap();
                    (tree, vanish_poly)
                } else {
                    let treen = load_tree(treenf, odd_leaf, num_constraints).unwrap();
                    // cached file should exist
                    let z_poly_coeffs = read_fr_vec_from_file(cache_dir.join(zpolyf))
                        .with_context(|| format!("expected pre‑computed {:?}", zpolyf))?;

                    if validate_precompute {
                        // Ensure downloaded vanishing polynomial was valid
                        let all_coeffs_zero =
                            z_poly_coeffs.iter().filter(|x| **x == Fr::zero()).count();
                        assert_ne!(
                            all_coeffs_zero,
                            z_poly_coeffs.len(),
                            "all polynomial coefficients were zero"
                        );
                        let evs = evaluate_vanishing_poly_at_domain(&z_poly_coeffs, &treen);
                        let not_zero_ev = evs.iter().find(|x| **x != Fr::zero());
                        assert!(
                            not_zero_ev.is_none(),
                            "vanishing poly does not evaluate to zero at all points in domain"
                        );
                    }

                    let vanish_poly = DensePolynomial {
                        coeffs: z_poly_coeffs,
                    };
                    (treen, vanish_poly)
                };

                let barycentric_weights = if cache_dir.join(barwtsf).exists() {
                    // read from cache if it exists
                    read_fr_vec_from_file(cache_dir.join(barwtsf)).unwrap()
                } else {
                    let barycentric_weights =
                        compute_barycentric_weights(&treen, &vanishing_poly).unwrap();
                    write_fr_vec_to_file(cache_dir.join(barwtsf), &barycentric_weights)?;
                    barycentric_weights
                };

                // instance(trapdoor) specific and used only for setup, so compute everytime
                let lag_basis = compute_lagrange_basis_at_tau(
                    &treen,
                    &vanishing_poly,
                    trapdoor.tau,
                    &barycentric_weights,
                )
                .unwrap();

                Ok((treen, lag_basis, vanishing_poly))
            };

        let (mut treen, l_tau, z_poly) =
            init_tree_and_poly_at_tau(TREE_2N, TREE_N, Z_POLY, BAR_WTS, false)?;

        // drop treen to save memory, will read from file in a couple of steps
        clear_fftree(&mut treen);

        let (mut treend, l_taud, z_polyd) =
            init_tree_and_poly_at_tau(TREE_2ND, TREE_ND, Z_POLYD, BAR_WTSD, true)?;

        // Pre‑compute domain vanishing polynomials and their inverses ---------
        let prepare_z_inv =
            |fname: &str, poly: &DensePolynomial<Fr>, domain: &FFTree<Fr>| -> Result<Vec<Fr>> {
                let path = cache_dir.join(fname);
                if path.exists() {
                    let v = read_fr_vec_from_file(&path)?;
                    Ok(v)
                } else {
                    let mut z_vals = evaluate_vanishing_poly_at_domain(poly, domain);
                    ark_ff::batch_inversion(&mut z_vals);
                    write_fr_vec_to_file(&path, &z_vals)?;
                    Ok(z_vals)
                }
            };

        println!("computing evaluations of vanishing polynomial on other domain");
        let mut z_vals2_inv = prepare_z_inv(Z_VALS2_INV, &z_poly, &treend)?;
        clear_fftree(&mut treend);

        // rebuild tree now that some space has been freed
        treen = if is_fresh_setup {
            let tree = load_tree(TREE_2N, false, num_constraints * 2).unwrap();
            let tree: FFTree<Fr> = tree.subtree_with_size(num_constraints).clone();
            tree
        } else {
            load_tree(TREE_N, false, num_constraints).unwrap()
        };

        println!("r1cs update_to_include_vandermode_matrix_d");
        R1CSInstance::update_to_include_vandermode_matrix_d(
            &mut inst,
            treen.f.leaves(),
            num_public_inputs,
        );

        let mut z_vals2d_inv = prepare_z_inv(Z_VALS2D_INV, &z_polyd, &treen)?;
        clear_fftree(&mut treen);

        println!("evaluate_lagrage_over_unified_domain_with_precompute");
        let l_taul = compute_lagrange_basis_at_tau_over_unified_domain(
            trapdoor.tau,
            num_constraints,
            &l_tau,
            &l_taud,
            &z_poly,
            &z_polyd,
            &z_vals2_inv,
            &z_vals2d_inv,
        );

        z_vals2_inv.clear();
        z_vals2d_inv.clear();

        println!("compute_srs_matrices");
        let srs_mats = compute_srs_matrices(
            cache_dir, &trapdoor, &z_poly, &mut inst, &l_tau, &l_taud, &l_taul,
        )?;

        Ok(Self {
            g_m: srs_mats.g_m,
            g_q: srs_mats.g_q,
            g_k: srs_mats.g_k,
        })
    }
}

impl SRS {
    pub(crate) fn empty() -> Self {
        SRS {
            g_m: vec![],
            g_q: vec![],
            g_k: [vec![], vec![], vec![]],
        }
    }

    /// verify
    pub fn verify(
        secrets: Trapdoor,
        public_inputs: &[Fr], // See: test_public_inputs_hash to understand how bridge public inputs will be passed to this function later
        proof: &Proof,
    ) -> bool {
        // Because winternitz signed data is compressed point form (30 bytes), we need to decompress it
        let (proof_commit_p, is_commit_p_valid) =
            CurvePoint::from_bytes(&mut proof.commit_p.clone());
        let (proof_kzg_k, is_kzg_k_valid) = CurvePoint::from_bytes(&mut proof.kzg_k.clone());

        let fs_challenge_alpha = {
            let mut transcript = Transcript::default();
            {
                // dummy srs
                let empty_srs = SRS {
                    g_m: vec![],
                    g_q: vec![],
                    g_k: [vec![], vec![], vec![]],
                };
                transcript.srs_hash(&empty_srs);
            }
            let inst = R1CSInstance {
                num_constraints: 0,
                num_public_inputs: 0,
                rows: vec![],
                coeffs: vec![],
            };

            transcript.circuit_info_hash(&inst);
            // The above two hashes is known at compile time and as such can be hardcoded
            // The following two has to be done in circuit

            transcript.public_input_hash(&public_inputs.to_vec());
            transcript.witness_commitment_hash(&[proof_commit_p]);

            transcript.output()
        };

        let i0 = evaluate_monomial_basis_poly(public_inputs, fs_challenge_alpha);

        let (proof_a0, is_a0_valid) = proof.a0.to_fr();
        let (proof_b0, is_b0_valid) = proof.b0.to_fr();
        let r0 = proof_a0 * proof_b0 - i0;
        // Step 3. Compute u₀ and v₀
        let delta2 = secrets.delta.square();
        let u0 = (proof_a0 + secrets.delta * proof_b0 + delta2 * r0) * secrets.epsilon;
        let v0 = (secrets.tau - fs_challenge_alpha) * secrets.epsilon;
        // Step 4. Check v₀·K == P - u₀·G;
        let lhs = multi_scalar_mul(&[v0, u0], &[proof_kzg_k, CurvePoint::generator()]);
        let rhs = proof_commit_p;

        let all_inputs_valid = is_a0_valid & is_b0_valid & is_commit_p_valid & is_kzg_k_valid;
        let valid_proof = lhs == rhs;
        valid_proof & all_inputs_valid
    }
}
