//! srs.rs
//!
use crate::artifacts::{
    BAR_WTS, BAR_WTSD, L_TAU, L_TAUD, L_TAUL, R1CS_CONSTRAINTS_FILE, SRS_G_K_0, SRS_G_K_1,
    SRS_G_K_2, SRS_G_M, SRS_G_Q, TREE_2N, TREE_2ND, TREE_N, TREE_ND, Z_POLY, Z_POLYD, Z_VALS2_INV,
    Z_VALS2D_INV,
};
use crate::curve::{CurvePoint, Fr, multi_scalar_mul, point_scalar_mul_gen};
use crate::ec_fft::{
    build_sect_ecfft_tree, evaluate_all_lagrange_coeffs_ecfft_with_vanish,
    evaluate_lagrage_over_unified_domain, evaluate_lagrage_over_unified_domain_with_precompute,
    evaluate_lagrange_coeffs_using_precompute, evaluate_vanishing_poly_over_domain,
    get_both_domains,
};
use crate::gnark_r1cs::{R1CSInstance, Row, load_sparse_r1cs_from_file};
use crate::io_utils::{
    read_fr_vec_from_file, read_point_vec_from_file, write_fr_vec_to_file, write_point_vec_to_file,
};
use crate::proving::{Proof, Transcript};
use crate::tree_io::{read_fftree_from_file, read_minimal_fftree_from_file, write_fftree_to_file};
use anyhow::{Context, Result};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_poly::Polynomial;
use ark_poly::univariate::DensePolynomial;
use ark_std::{rand::Rng, vec::Vec};
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

// --- Helper Function to Abstract File I/O and Computation ---
// This function handles the common pattern of reading data from a file if it exists,
// or computing it, writing it to the file, and then returning it.

/// Reads a vector of `CurvePoint` from `file_path` if it exists; otherwise
/// computes it via `compute_fn`, stores it, and returns the freshly computed
/// value. This removes a lot of repetitive I/O boilerplate.
fn compute_or_read_vec<F>(file_path: impl AsRef<Path>, compute_fn: F) -> Result<Vec<CurvePoint>>
where
    F: FnOnce() -> Result<Vec<CurvePoint>>, // computation may fail too
{
    let file_path = file_path.as_ref();

    if file_path.exists() {
        println!("Reading {:?} …", file_path.display());
        read_point_vec_from_file(file_path)
            .with_context(|| format!("reading {:?}", file_path.display()))
    } else {
        println!("Computing {:?} …", file_path.display());
        let result = compute_fn()?;
        write_point_vec_to_file(file_path, &result)
            .with_context(|| format!("writing {:?}", file_path.display()))?;
        Ok(result)
    }
}

/// Builds the SRS matrices required by the verifier.  All intermediate artefacts
/// are cached on disk under `cache_dir` so that subsequent runs can skip
/// expensive recomputation.
fn compute_srs_matrices(
    cache_dir: &Path,
    secrets: &Trapdoor,
    z_poly_ark: &DensePolynomial<Fr>,
    inst: &R1CSInstance,
    l_tau: &[Fr],
    l_taud: &[Fr],
    l_taul: &[Fr],
) -> Result<SRSMatrices> {
    let (tau, delta) = (secrets.tau, secrets.delta);
    let z_tau = z_poly_ark.evaluate(&tau);
    let delta2 = delta.square();

    // g_m -------------------------------------------------------------------
    let g_m = compute_or_read_vec(cache_dir.join(SRS_G_M), || {
        let m_vals = accumulate_m_values(&inst.rows, &inst.coeffs, l_tau, delta);
        Ok(m_vals
            .into_par_iter()
            .map(|val| point_scalar_mul_gen(val * secrets.epsilon))
            .collect())
    })?;

    // g_q -------------------------------------------------------------------
    let g_q = compute_or_read_vec(cache_dir.join(SRS_G_Q), || {
        Ok((0..inst.num_constraints)
            .into_par_iter()
            .map(|i| {
                let scalar = z_tau * delta2 * l_taud[i] * secrets.epsilon;
                point_scalar_mul_gen(scalar)
            })
            .collect())
    })?;

    // g_k vectors ------------------------------------------------------------
    let delta_pows = [Fr::one(), delta, delta2];
    let mut g_k_vecs: [Vec<CurvePoint>; 3] = Default::default();
    const SRS_GK: [&str; 3] = [SRS_G_K_0, SRS_G_K_1, SRS_G_K_2];
    for (j, g_k) in g_k_vecs.iter_mut().enumerate() {
        let path = cache_dir.join(SRS_GK[j]);
        let l_slice = if j < 2 { l_tau } else { l_taul };

        *g_k = compute_or_read_vec(path, || {
            Ok(l_slice
                .into_par_iter()
                .map(|l_val| point_scalar_mul_gen(*l_val * delta_pows[j]))
                .collect())
        })?;
    }

    Ok(SRSMatrices {
        g_m,
        g_q,
        g_k: g_k_vecs,
    })
}

impl SRS {
    /// verifier_runs_fresh_setup
    pub fn verifier_runs_fresh_setup<R: Rng>(
        rng: &mut R,
        cache_dir: &Path,
        num_public_inputs: usize,
    ) -> Result<(Self, Trapdoor)> {
        std::fs::create_dir_all(cache_dir) // ensure directory exists
            .with_context(|| format!("creating {}", cache_dir.display()))?;

        let dump =
            load_sparse_r1cs_from_file(File::open(cache_dir.join(R1CS_CONSTRAINTS_FILE)).unwrap())
                .unwrap();
        let mut inst = R1CSInstance::from_dump(dump.clone(), num_public_inputs);

        let num_constraints = inst.num_constraints;
        let n_log = num_constraints.ilog2() as usize;

        // Trapdoor secrets ---------------------------------------------------
        let tau = Fr::rand(rng);
        let delta = Fr::rand(rng);
        let epsilon = Fr::rand(rng);

        assert!(!epsilon.is_zero(), "ε must be non‑zero");

        // Helper: read‑or‑build an ECFFT tree --------------------------------
        let load_tree = |name: &str, odd_leaves: bool| -> Result<FFTree<Fr>> {
            let path = cache_dir.join(name);
            if path.exists() {
                println!("Reading {:?} …", path.display());
                read_fftree_from_file(&path)
            } else {
                println!("Computing {:?} …", path.display());
                let tree = build_sect_ecfft_tree(num_constraints * 2, odd_leaves, n_log + 1, false)
                    .unwrap();
                write_fftree_to_file(&tree, &path)?;
                Ok(tree)
            }
        };

        println!("Building ECFFT trees …");
        let mut tree2n = load_tree(TREE_2N, false)?;

        // l_tau and vanishing polynomial over τ ------------------------------
        let (l_tau, z_poly) = {
            let path_l = cache_dir.join(L_TAU);
            let path_z = cache_dir.join(Z_POLY);

            if path_l.exists() {
                (
                    read_fr_vec_from_file(&path_l)?,
                    read_fr_vec_from_file(&path_z)?,
                )
            } else {
                let res = evaluate_all_lagrange_coeffs_ecfft_with_vanish(&tree2n, tau).unwrap();
                let (lag_basis, vanish_poly_coeff, barycentric_weights) = (
                    res.lagrange_coeffs,
                    res.z_s_coeffs_vec_from_exit,
                    res.z_s_prime_evals_on_s,
                );
                write_fr_vec_to_file(&path_l, &lag_basis)?;
                write_fr_vec_to_file(&path_z, &vanish_poly_coeff)?;
                write_fr_vec_to_file(cache_dir.join(BAR_WTS), &barycentric_weights)?;
                (lag_basis, vanish_poly_coeff)
            }
        };
        let treen = tree2n.subtree_with_size(inst.num_constraints).clone();
        clear_fftree(&mut tree2n);

        R1CSInstance::update_to_include_vandermode_matrix_d(
            &mut inst,
            treen.f.leaves(),
            num_public_inputs,
        );

        // l_taud and vanishing polynomial over τ·δ ---------------------------
        let mut tree2nd = load_tree(TREE_2ND, true)?;
        let (l_taud, z_polyd) = {
            let path_l = cache_dir.join(L_TAUD);
            let path_z = cache_dir.join(Z_POLYD);

            if path_l.exists() {
                (
                    read_fr_vec_from_file(&path_l)?,
                    read_fr_vec_from_file(&path_z)?,
                )
            } else {
                let res = evaluate_all_lagrange_coeffs_ecfft_with_vanish(&tree2nd, tau).unwrap();
                let (lag_basis, vanish_poly_coeff, barycentric_weights) = (
                    res.lagrange_coeffs,
                    res.z_s_coeffs_vec_from_exit,
                    res.z_s_prime_evals_on_s,
                );
                write_fr_vec_to_file(&path_l, &lag_basis)?;
                write_fr_vec_to_file(&path_z, &vanish_poly_coeff)?;
                write_fr_vec_to_file(cache_dir.join(BAR_WTSD), &barycentric_weights)?;
                (lag_basis, vanish_poly_coeff)
            }
        };

        let treend = tree2nd.subtree_with_size(inst.num_constraints).clone();
        clear_fftree(&mut tree2nd);

        // l_taul -------------------------------------------------------------
        let z_poly_ark = DensePolynomial { coeffs: z_poly };
        let l_taul = {
            let path = cache_dir.join(L_TAUL);

            if path.exists() {
                read_fr_vec_from_file(&path)?
            } else {
                let lag_basis = evaluate_lagrage_over_unified_domain(
                    tau,
                    &treen,
                    &l_tau,
                    &z_poly_ark.coeffs,
                    &treend,
                    &l_taud,
                    &z_polyd,
                );
                write_fr_vec_to_file(&path, &lag_basis)?;
                lag_basis
            }
        };

        // Pre‑compute domain vanishing polynomials and their inverses ---------
        let prepare_z_inv =
            |fname: &str, poly: &DensePolynomial<Fr>, domain: &FFTree<Fr>| -> Result<Vec<Fr>> {
                let path = cache_dir.join(fname);
                if path.exists() {
                    let mut z_inv = read_fr_vec_from_file(&path)?;
                    ark_ff::batch_inversion(&mut z_inv);
                    Ok(z_inv)
                } else {
                    let mut z_vals = evaluate_vanishing_poly_over_domain(poly, domain);
                    let z_clone = z_vals.clone();
                    ark_ff::batch_inversion(&mut z_vals);
                    write_fr_vec_to_file(&path, &z_vals)?;
                    Ok(z_clone)
                }
            };

        // println!("Preparing z‑inv tables …");
        let _ = prepare_z_inv(Z_VALS2_INV, &z_poly_ark, &treend)?;
        let _ = prepare_z_inv(
            Z_VALS2D_INV,
            &DensePolynomial {
                coeffs: z_polyd.clone(),
            },
            &treen,
        )?;

        let trapdoor = Trapdoor {
            tau,
            delta,
            epsilon,
        };

        let srs_mats = compute_srs_matrices(
            cache_dir,
            &trapdoor,
            &z_poly_ark,
            &inst,
            &l_tau,
            &l_taud,
            &l_taul,
        )?;

        Ok((
            Self {
                g_m: srs_mats.g_m,
                g_q: srs_mats.g_q,
                g_k: srs_mats.g_k,
            },
            trapdoor,
        ))
    }

    /// verifier_runs_setup_with_precompute
    pub fn verifier_runs_setup_with_precompute<R: Rng>(
        rng: &mut R,
        cache_dir: impl AsRef<Path>,
        num_public_inputs: usize,
    ) -> Result<(Self, Trapdoor)> {
        let cache_dir = cache_dir.as_ref();
        std::fs::create_dir_all(cache_dir)
            .with_context(|| format!("creating {}", cache_dir.display()))?;

        let dump =
            load_sparse_r1cs_from_file(File::open(cache_dir.join(R1CS_CONSTRAINTS_FILE)).unwrap())
                .unwrap();
        let mut inst = R1CSInstance::from_dump(dump.clone(), num_public_inputs);

        let num_constraints = inst.num_constraints;
        let n_log = num_constraints.ilog2() as usize;

        // Trapdoor secrets ---------------------------------------------------
        let tau = Fr::rand(rng);
        let delta = Fr::rand(rng);
        let epsilon = Fr::rand(rng);
        assert!(!epsilon.is_zero(), "ε must be non-zero");

        // Helper: read-or-build an ECFFT tree for *exact* domain size ---------
        let load_tree = |name: &str, shift_by_one: bool| -> Result<FFTree<Fr>> {
            let path = cache_dir.join(name);
            if path.exists() {
                println!("Reading {:?} …", path.display());
                read_fftree_from_file(&path)
            } else {
                println!("Computing {:?} …", path.display());
                let tree =
                    build_sect_ecfft_tree(num_constraints, shift_by_one, n_log + 1, false).unwrap();
                write_fftree_to_file(&tree, &path)?;
                Ok(tree)
            }
        };

        let mut treen = load_tree(TREE_N, false)?;

        R1CSInstance::update_to_include_vandermode_matrix_d(
            &mut inst,
            treen.f.leaves(),
            num_public_inputs,
        );

        // ------------ Helper: read or abort if missing (precompute assumed) --
        let must_read = |fname: &str| -> Result<Vec<Fr>> {
            let path = cache_dir.join(fname);
            read_fr_vec_from_file(&path)
                .with_context(|| format!("expected pre‑computed {:?}", path.display()))
        };

        let z_poly = must_read(Z_POLY)?;
        let mut barycentric_weights = must_read(BAR_WTS)?;

        // l_tau using existing precompute ------------------------------------
        let l_tau = {
            let path = cache_dir.join(L_TAU);
            if path.exists() {
                read_fr_vec_from_file(&path)?
            } else {
                let lag_basis = evaluate_lagrange_coeffs_using_precompute(
                    &treen,
                    tau,
                    z_poly.clone(),
                    &mut barycentric_weights,
                )
                .unwrap();
                write_fr_vec_to_file(&path, &lag_basis)?;
                lag_basis
            }
        };

        barycentric_weights.clear();
        clear_fftree(&mut treen);

        // z_polyd + bar_wtsd --------------------------------------------------
        let mut treend = load_tree(TREE_ND, true)?;

        let z_polyd = must_read(Z_POLYD)?;
        let mut barycentric_weightsd = must_read(BAR_WTSD)?;

        // l_taud -------------------------------------------------------------
        let l_taud = {
            let path = cache_dir.join(L_TAUD);
            if path.exists() {
                read_fr_vec_from_file(&path)?
            } else {
                let lag_basis = evaluate_lagrange_coeffs_using_precompute(
                    &treend,
                    tau,
                    z_polyd.clone(),
                    &mut barycentric_weightsd,
                )
                .unwrap();
                write_fr_vec_to_file(&path, &lag_basis)?;
                lag_basis
            }
        };
        clear_fftree(&mut treend);
        barycentric_weights.clear();

        let z_poly_ark = DensePolynomial { coeffs: z_poly };
        let mut z_vals2_inv = must_read(Z_VALS2_INV)?;
        let mut z_vals2d_inv = must_read(Z_VALS2D_INV)?;

        // l_taul -------------------------------------------------------------
        let l_taul = {
            let path = cache_dir.join(L_TAUL);
            if path.exists() {
                read_fr_vec_from_file(&path)?
            } else {
                let lag_basis = evaluate_lagrage_over_unified_domain_with_precompute(
                    tau,
                    num_constraints,
                    &l_tau,
                    &l_taud,
                    &z_poly_ark.coeffs,
                    &z_polyd,
                    &z_vals2_inv,
                    &z_vals2d_inv,
                );
                write_fr_vec_to_file(&path, &lag_basis)?;
                lag_basis
            }
        };
        z_vals2_inv.clear();
        z_vals2d_inv.clear();

        // --------------------------------------------------------------------
        let trapdoor = Trapdoor {
            tau,
            delta,
            epsilon,
        };

        let srs_mats = compute_srs_matrices(
            cache_dir,
            &trapdoor,
            &z_poly_ark,
            &inst,
            &l_tau,
            &l_taud,
            &l_taul,
        )?;

        Ok((
            Self {
                g_m: srs_mats.g_m,
                g_q: srs_mats.g_q,
                g_k: srs_mats.g_k,
            },
            trapdoor,
        ))
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
        &self,
        cache_dir: &str,
        secrets: Trapdoor,
        public_inputs: &[Fr], // See: test_public_inputs_hash to understand how bridge public inputs will be passed to this function later
        proof: &Proof,
    ) -> bool {
        println!("proof {:?}", proof);
        println!("trapdoor {:?}", secrets);

        // Because winternitz signed data is compressed point form (30 bytes), we need to decompress it
        let proof_commit_p = CurvePoint::from_bytes(&mut proof.commit_p.clone());
        let proof_kzg_k = CurvePoint::from_bytes(&mut proof.kzg_k.clone());

        const G_KS: [&str; 3] = [SRS_G_K_0, SRS_G_K_1, SRS_G_K_2];
        let fs_challenge_alpha = {
            let mut transcript = Transcript::default();
            {
                let g_m = read_point_vec_from_file(format!("{cache_dir}/{SRS_G_M}")).unwrap();
                let g_q = read_point_vec_from_file(format!("{cache_dir}/{SRS_G_Q}")).unwrap();
                let g_k: Vec<Vec<CurvePoint>> = (0..3)
                    .map(|i| read_point_vec_from_file(format!("{cache_dir}/{}", G_KS[i])).unwrap())
                    .collect();
                // also include g_k4
                let srs = SRS {
                    g_m,
                    g_q,
                    g_k: g_k.try_into().unwrap(),
                };
                transcript.srs_hash(&srs);
            }
            let tree2n: FFTree<Fr> =
                read_minimal_fftree_from_file(Path::new(&format!("{cache_dir}/{TREE_2N}")))
                    .unwrap();

            let doms = &get_both_domains(&tree2n)[0];

            let inst = R1CSInstance::initialize_with_vandermode_matrix_for_public_input_poly(
                &format!("{cache_dir}/{R1CS_CONSTRAINTS_FILE}"),
                doms,
                public_inputs.len(),
            );

            transcript.circuit_info_hash(&inst);
            // The above two hashes is known at compile time and as such can be hardcoded
            // The following two has to be done in circuit

            transcript.public_input_hash(&public_inputs.to_vec());
            transcript.witness_commitment_hash(&[proof_commit_p]);

            transcript.output()
        };

        let i0 = public_inputs[1] * fs_challenge_alpha + public_inputs[0];

        let r0 = proof.a0 * proof.b0 - i0;
        // Step 3. Compute u₀ and v₀
        let delta2 = secrets.delta.square();
        let u0 = (proof.a0 + secrets.delta * proof.b0 + delta2 * r0) * secrets.epsilon;
        let v0 = (secrets.tau - fs_challenge_alpha) * secrets.epsilon;
        // Step 4. Check v₀·K == P - u₀·G;
        let lhs = multi_scalar_mul(&[v0, u0], &[proof_kzg_k, CurvePoint::generator()]);
        let rhs = proof_commit_p;
        lhs == rhs
    }
}
