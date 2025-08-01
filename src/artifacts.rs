// File includes constant names
/// We have artifacts of three types
/// SRS - G_Q, G_M, G_K, L_tau* (Should be computed)
/// MainPrecompute - Z_poly, Z_polyd (Recommended to download)
/// AuxPrecompute = bar_wts, z_valsinv, tree2n_fast,
pub(crate) const SRS_G_Q: &str = "g_q";
pub(crate) const SRS_G_M: &str = "g_m";
pub(crate) const SRS_G_K_0: &str = "g_k_0";
pub(crate) const SRS_G_K_1: &str = "g_k_1";
pub(crate) const SRS_G_K_2: &str = "g_k_2";

pub(crate) const TREE_2N: &str = "tree2n_fast";
pub(crate) const TREE_2ND: &str = "tree2nd_fast";
pub(crate) const TREE_N: &str = "treen_fast";
pub(crate) const TREE_ND: &str = "treend_fast";

pub(crate) const L_TAU: &str = "l_tau";
pub(crate) const Z_POLY: &str = "z_poly";
pub(crate) const L_TAUD: &str = "l_taud";
pub(crate) const Z_POLYD: &str = "z_polyd";

pub(crate) const BAR_WTS: &str = "bar_wts";
pub(crate) const BAR_WTSD: &str = "bar_wtsd";

pub(crate) const L_TAUL: &str = "l_taul";

pub(crate) const Z_VALS2_INV: &str = "z_vals2inv";
pub(crate) const Z_VALS2D_INV: &str = "z_vals2dinv";

pub(crate) const R1CS_CONSTRAINTS_FILE: &str = "r1cs_temp";
#[cfg(test)]
pub(crate) const R1CS_WITNESS_FILE: &str = "witness_temp";

/*
Verifier

| File | Download | Compute | Age | Why Cache It ? |
| -- | -- | -- | -- | -- |
| Tree2n | 15 GB, shasum | 20 mins | same num_constraints | efficient |
| Tree2nd | 15 GB, shasum | 20 mins | same num_constraints | efficient |
| z_poly | 200 MB, verifiable | 2 hrs+ | same num_constraints | efficient |
| z_polyd | 200 MB, verifiable | 2 hrs+ | same num_constraints | efficient |

| z_vals2inv | 200 MB, shasum | 10 mins | same num_constraints | dev/tests/share |

| l_tau | - | 20 mins | same trapdoor | dev/tests
| G_Q, G_M, G_K | - | 10 mins | same trapdoor | share with prover |

| r1cs | 800 MB, shasum | 20 mins | received from external code |
| witness | 200 MB, shasum | 20 mins | received from external code |

*/

/*
Prover

| File | Download | Compute | Age | Why Cache It ? |
| -- | -- | -- | -- | -- |
| Tree2n_mini | 7 GB, shasum | 10 mins | same num_constraints | efficient |
| z_poly | 200 MB, verifiable | 2 hrs+ | same num_constraints | efficient |
| z_polyd | 200 MB, verifiable | 2 hrs+ | same num_constraints | efficient |

| z_vals2inv | 200 MB, shasum | 10 mins | same num_constraints | dev/tests/share |
| bar_wts | 200 MB, shasum | 10 mins | same num_constraints | dev/tests/share |

| l_tau | - | 20 mins | same trapdoor | dev/tests
| G_Q, G_M, G_K | - | 10 mins | same trapdoor | share with prover |

| r1cs | 800 MB, shasum | 20 mins | received from external code |
| witness | 200 MB, shasum | 20 mins | received from external code |

*/
