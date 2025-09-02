// input is srs
// this includes g_q, g_m, g_k

// sample w, q and compute commitment
// this is equivalent to generating proof
// he will compute P, K

use crate::{
    curve::{
        CompressedCurvePoint, CurvePoint, Fr, multi_scalar_mul, point_scalar_mul,
        point_scalar_mul_gen,
    },
    gnark_r1cs::R1CSInstance,
    proving::{Proof, Transcript},
    srs::{SRS, Trapdoor},
};
use ark_ff::Field;
use ark_std::rand::SeedableRng;

#[derive(Clone, Copy)]
pub struct TrapdoorBlindingFactors {
    pub bf_tau: Fr,
    pub bf_delta: Fr,
    pub bf_epsilon: Fr,
}

pub struct TrapdoorCommitments {
    commit_r0: CurvePoint,
    commit_r1: CurvePoint,
    commit_r2: CurvePoint,
    commit_r_tau: CurvePoint,
    commit_r_delta: CurvePoint,
    commit_r_epsilon: CurvePoint,

    commit_h_tau: CurvePoint,
    commit_h_delta: CurvePoint,
    commit_h_epsilon: CurvePoint,
}

pub(crate) fn srsverifier_commits_to_witness(
    cache_dir: &str,
    witness: &[Fr],
    num_public_inputs: usize,
) -> (Proof, Fr) {
    assert_eq!(witness[0], Fr::ONE);
    let proof = Proof::prove(
        cache_dir,
        witness[1..1 + num_public_inputs].to_vec(),
        &witness[1 + num_public_inputs..],
    );

    let (proof_commit_p, is_commit_p_valid) = CurvePoint::from_bytes(&mut proof.commit_p.clone());
    assert!(is_commit_p_valid);

    let fs_challenge_alpha = {
        let mut transcript = Transcript::default();
        let empty_srs = SRS {
            g_m: vec![],
            g_q: vec![],
            g_k: [vec![], vec![], vec![]],
        };
        let inst = R1CSInstance {
            num_constraints: 0,
            num_public_inputs: 0,
            rows: vec![],
            coeffs: vec![],
        };
        transcript.srs_hash(&empty_srs);
        transcript.circuit_info_hash(&inst);
        transcript.public_input_hash(&witness[1..1 + num_public_inputs].to_vec());
        transcript.witness_commitment_hash(&[proof_commit_p]);

        transcript.output()
    };

    (proof, fs_challenge_alpha)
}

// fn srsprover_samples_trapdoor_blindings() -> TrapdoorBlindingFactors;

/// srsprover_computes_commitments
// R_0 + R_1 Y + R_2 Y^2 = Y^2 P - [(aY + b d(Y)) e(Y)]G - [(t(Y) - aY) e(Y)]Q
// => R_0 + R_1 Y + R_2 Y^2  = Y^2 P - [( (aY + b (r_d + Y d) ) (r_e + Y e) ] G - [ ( (r_t + Y t) - aY) (r_e + Y e) ] Q
// =>
// R_0  = [b r_d r_e] G + [r_t r_e] Q
// R_1 = [a r_e + b d r_e + b r_d e] G + [t r_e - alpha r_e + e r_t] Q
// R_2 = P + [a e + b d e] G + [t e - alpha e] Q
pub(crate) fn srsprover_computes_commitments(
    trapdoor: Trapdoor,
    trapdoor_bf: TrapdoorBlindingFactors,
    proof: Proof,
    proof_alpha: Fr,
    gen_h: CurvePoint,
) -> TrapdoorCommitments {
    let (proof_a0, is_a0_valid) = proof.a0.to_fr();
    let (proof_b0, is_b0_valid) = proof.b0.to_fr();
    assert!(is_a0_valid, "a0 should be valid");
    assert!(is_b0_valid, "b0 should be valid");

    let (proof_commit_p, is_commit_p_valid) = CurvePoint::from_bytes(&mut proof.commit_p.clone());
    assert!(is_commit_p_valid, "P should be valid");
    let (proof_kzg_k, is_kzg_k_valid) = CurvePoint::from_bytes(&mut proof.kzg_k.clone());
    assert!(is_kzg_k_valid, "K should be valid");

    let r_0_x = proof_b0 * trapdoor_bf.bf_delta * trapdoor_bf.bf_epsilon;
    let r_0_y = trapdoor_bf.bf_tau * trapdoor_bf.bf_epsilon;

    let r_1_x = proof_a0 * trapdoor_bf.bf_epsilon
        + proof_b0 * trapdoor.delta * trapdoor_bf.bf_epsilon
        + proof_b0 * trapdoor_bf.bf_delta * trapdoor.epsilon;
    let r_1_y = trapdoor.tau * trapdoor_bf.bf_epsilon - proof_alpha * trapdoor_bf.bf_epsilon
        + trapdoor.epsilon * trapdoor_bf.bf_tau;

    let r_2_x = proof_a0 * trapdoor.epsilon + proof_b0 * trapdoor.delta * trapdoor.epsilon;
    let r_2_y = trapdoor.tau * trapdoor.epsilon - proof_alpha * trapdoor.epsilon;

    let commit_r0 = multi_scalar_mul(&[-r_0_x, -r_0_y], &[CurvePoint::generator(), proof_kzg_k]);
    let commit_r1 = multi_scalar_mul(&[-r_1_x, -r_1_y], &[CurvePoint::generator(), proof_kzg_k]);
    let commit_r2 = CurvePoint::add(
        proof_commit_p,
        multi_scalar_mul(&[-r_2_x, -r_2_y], &[CurvePoint::generator(), proof_kzg_k]),
    );

    let commit_r_tau = point_scalar_mul(trapdoor_bf.bf_tau, gen_h);
    let commit_r_delta = point_scalar_mul(trapdoor_bf.bf_delta, gen_h);
    let commit_r_epsilon = point_scalar_mul(trapdoor_bf.bf_epsilon, gen_h);

    let commit_h_tau = point_scalar_mul(trapdoor.tau, gen_h);
    let commit_h_delta = point_scalar_mul(trapdoor.delta, gen_h);
    let commit_h_epsilon = point_scalar_mul(trapdoor.epsilon, gen_h);

    TrapdoorCommitments {
        commit_r0,
        commit_r1,
        commit_r2,
        commit_r_tau,
        commit_r_delta,
        commit_r_epsilon,
        commit_h_delta,
        commit_h_epsilon,
        commit_h_tau,
    }
}

//fn srsverifier_samples_challenge() -> Fr;

pub struct ChallengeResponse {
    ch_tau: Fr,
    ch_delta: Fr,
    ch_epsilon: Fr,
}

pub(crate) fn srsprover_computes_challenge_response(
    challenge: Fr,
    trapdoor: Trapdoor,
    trapdoor_bf: TrapdoorBlindingFactors,
) -> ChallengeResponse {
    ChallengeResponse {
        ch_tau: trapdoor_bf.bf_tau + challenge * trapdoor.tau,
        ch_delta: trapdoor_bf.bf_delta + challenge * trapdoor.delta,
        ch_epsilon: trapdoor_bf.bf_epsilon + challenge * trapdoor.epsilon,
    }
}

pub(crate) fn srsverifier_verifies_commitment(
    challenge: Fr,
    challenge_resp: ChallengeResponse,
    gen_h: CurvePoint,
    commits: TrapdoorCommitments,
    proof: Proof,
    proof_alpha: Fr,
) -> bool {
    let (proof_a0, is_a0_valid) = proof.a0.to_fr();
    let (proof_b0, is_b0_valid) = proof.b0.to_fr();
    assert!(is_a0_valid, "a0 should be valid");
    assert!(is_b0_valid, "b0 should be valid");

    let (proof_commit_p, is_commit_p_valid) = CurvePoint::from_bytes(&mut proof.commit_p.clone());
    assert!(is_commit_p_valid, "P should be valid");
    let (proof_kzg_k, is_kzg_k_valid) = CurvePoint::from_bytes(&mut proof.kzg_k.clone());
    assert!(is_kzg_k_valid, "K should be valid");

    let lhs = point_scalar_mul(challenge_resp.ch_tau, gen_h);
    let rhs = CurvePoint::add(
        commits.commit_r_tau,
        point_scalar_mul(challenge, commits.commit_h_tau),
    );
    let matches = lhs == rhs;

    let lhs_x = point_scalar_mul(challenge, commits.commit_r1);
    let lhs_y = point_scalar_mul(challenge * challenge, commits.commit_r2);
    let lhs = CurvePoint::add(CurvePoint::add(commits.commit_r0, lhs_x), lhs_y);

    let rhs_x = point_scalar_mul(challenge * challenge, proof_commit_p);
    let rhs_y = point_scalar_mul_gen(
        -(proof_a0 * challenge + challenge_resp.ch_delta * proof_b0) * challenge_resp.ch_epsilon,
    );
    let rhs_z = point_scalar_mul(
        -(challenge_resp.ch_tau - proof_alpha * challenge) * challenge_resp.ch_epsilon,
        proof_kzg_k,
    );
    let rhs = CurvePoint::add(CurvePoint::add(rhs_x, rhs_y), rhs_z);

    let matches2 = lhs == rhs;
    matches & matches2
}
