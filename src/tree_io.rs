//! High-performance I/O for ecfft::fftree::FFTree
//!
//! Format (little-endian):
//! ┌── FileHeader (8 B magic + 8 B total_len) ──────────────────────────┐
//! │ b"FFTR\0\0\0\0" │ u64 total_bytes_of_all_nodes                     │
//! └────────────────────────────────────────────────────────────────────┘
//! then a **flat, depth-first list** of nodes; each node =
//!   NodeHeader { u32 section_count; u32 _pad }               (8  B)
//!   SectionMeta[section_count] (id:u8 + 7 pad + u64 off + len) (24 B ea)
//!   all blobs for this node (compressed ark-serialize output)
//!
//! Section IDs 0-11 are the big vectors inside one node;
//! ID 12 (Subtree) is, if present, **one complete child FFTree blob**
//! written with the *same* layout.  Deserialiser recurses and spawns
//! Rayon tasks at every level.

use anyhow::{Result, bail};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use ecfft::{
    fftree::FFTree,
    utils::{BinaryTree, Mat2x2, RationalMap}, // adjust if your paths differ
};
use memmap2::Mmap;
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};

// ────────────────────── per-node section IDs ────────────────────────────
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SectionId {
    FLeaves = 0,
    RecombineMatrices = 1,
    DecomposeMatrices = 2,
    RationalMaps = 3,
    XnnS = 4,
    XnnSInv = 5,
    Z0S1 = 6,
    Z1S0 = 7,
    Z0InvS1 = 8,
    Z1InvS0 = 9,
    Z0Z0RemXnnS = 10,
    Z1Z1RemXnnS = 11,
    Subtree = 12,
}

impl TryFrom<u8> for SectionId {
    type Error = anyhow::Error;
    fn try_from(v: u8) -> Result<Self> {
        use SectionId::*;
        Ok(match v {
            0 => FLeaves,
            1 => RecombineMatrices,
            2 => DecomposeMatrices,
            3 => RationalMaps,
            4 => XnnS,
            5 => XnnSInv,
            6 => Z0S1,
            7 => Z1S0,
            8 => Z0InvS1,
            9 => Z1InvS0,
            10 => Z0Z0RemXnnS,
            11 => Z1Z1RemXnnS,
            12 => Subtree,
            _ => bail!("unknown section id {v}"),
        })
    }
}

// ─────────────────────── fixed-width headers ───────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
struct NodeHeader {
    section_count: u32,
    _pad: u32,
}
impl NodeHeader {
    fn to_bytes(self) -> [u8; 8] {
        let mut o = [0u8; 8];
        o[..4].copy_from_slice(&self.section_count.to_le_bytes());
        o
    }
    fn from_slice(s: &[u8]) -> Self {
        Self {
            section_count: u32::from_le_bytes(s[..4].try_into().unwrap()),
            _pad: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SectionMeta {
    id: u8,
    _pad: [u8; 7],
    off: u64,
    len: u64,
}
impl SectionMeta {
    fn to_bytes(self) -> [u8; 24] {
        let mut o = [0u8; 24];
        o[0] = self.id;
        o[8..16].copy_from_slice(&self.off.to_le_bytes());
        o[16..24].copy_from_slice(&self.len.to_le_bytes());
        o
    }
    fn from_slice(s: &[u8]) -> Self {
        Self {
            id: s[0],
            _pad: [0; 7],
            off: u64::from_le_bytes(s[8..16].try_into().unwrap()),
            len: u64::from_le_bytes(s[16..24].try_into().unwrap()),
        }
    }
}

// ─────────────── helper macros (define *before* use) ────────────────────
macro_rules! ser {
    ($f:expr) => {{
        let mut v = Vec::<u8>::new();
        $f.serialize_compressed(&mut v)?;
        v
    }};
}
macro_rules! de {
    ($t:ty, $bytes:expr) => {
        <$t>::deserialize_with_mode($bytes, Compress::Yes, Validate::No)
    };
}
macro_rules! slice {
    ($secs:expr, $sid:expr) => {{
        $secs
            .iter()
            .find(|(id, _)| *id == $sid)
            .map(|(_, s)| *s)
            .expect("missing section")
    }};
}

/// Write [`ecfft::fftree::FFTree`] to file
pub(crate) fn write_fftree_to_file<F: Field + CanonicalSerialize>(
    tree: &FFTree<F>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let bytes = write_fftree_to_vec(tree)?;
    const MAGIC: &[u8; 8] = b"FFTR\0\0\0\0";
    let mut w = BufWriter::with_capacity(
        1 << 20,
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?,
    );
    w.write_all(MAGIC)?;
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

// recursive helper → Vec<u8>
fn write_fftree_to_vec<F: Field + CanonicalSerialize>(n: &FFTree<F>) -> Result<Vec<u8>> {
    use SectionId::*;
    // ─ collect blobs
    let mut blobs = vec![
        (FLeaves, ser!(n.f)),
        (RecombineMatrices, ser!(n.recombine_matrices)),
        (DecomposeMatrices, ser!(n.decompose_matrices)),
        (RationalMaps, ser!(n.rational_maps)),
        (XnnS, ser!(n.xnn_s)),
        (XnnSInv, ser!(n.xnn_s_inv)),
        (Z0S1, ser!(n.z0_s1)),
        (Z1S0, ser!(n.z1_s0)),
        (Z0InvS1, ser!(n.z0_inv_s1)),
        (Z1InvS0, ser!(n.z1_inv_s0)),
        (Z0Z0RemXnnS, ser!(n.z0z0_rem_xnn_s)),
        (Z1Z1RemXnnS, ser!(n.z1z1_rem_xnn_s)),
    ];
    if let Some(ref sub) = n.subtree {
        blobs.push((Subtree, write_fftree_to_vec(sub)?));
    }

    // ─ lay out header + table
    let mut out = Vec::<u8>::new();
    out.extend_from_slice(
        &NodeHeader {
            section_count: blobs.len() as u32,
            _pad: 0,
        }
        .to_bytes(),
    );
    let mut metas = Vec::<SectionMeta>::with_capacity(blobs.len());
    let mut cur = 8 + 24 * blobs.len();
    for (id, b) in &blobs {
        metas.push(SectionMeta {
            id: *id as u8,
            _pad: [0; 7],
            off: cur as u64,
            len: b.len() as u64,
        });
        cur += b.len();
    }
    for m in &metas {
        out.extend_from_slice(&m.to_bytes());
    }
    for (_, b) in &blobs {
        out.extend_from_slice(b);
    }
    Ok(out)
}

/// Read [`ecfft::fftree::FFTree`] from file
pub(crate) fn read_fftree_from_file<F>(path: impl AsRef<Path>) -> Result<FFTree<F>>
where
    F: Field + CanonicalDeserialize + Send + Sync,
{
    const MAGIC: &[u8; 8] = b"FFTR\0\0\0\0";
    let f = File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    if &mmap[..8] != MAGIC {
        bail!("not an FFTR file");
    }
    let total = u64::from_le_bytes(mmap[8..16].try_into()?) as usize;
    let slice = &mmap[16..16 + total];
    let (tree, _) = read_fftree_from_slice::<F>(slice)?;
    Ok(tree)
}

// pub(crate) fn read_fftree_from_file<F>(path: impl AsRef<Path>) -> Result<FFTree<F>> {
// }

// recursive read-helper; returns (tree, bytes_consumed)
fn read_fftree_from_slice<F>(bytes: &[u8]) -> Result<(FFTree<F>, usize)>
where
    F: Field + CanonicalDeserialize + Send + Sync,
{
    use SectionId::*;

    // ─ header + table
    let hdr = NodeHeader::from_slice(&bytes[..8]);
    let mut metas = Vec::<SectionMeta>::with_capacity(hdr.section_count as usize);
    let mut pos = 8;
    for _ in 0..hdr.section_count {
        metas.push(SectionMeta::from_slice(&bytes[pos..pos + 24]));
        pos += 24;
    }
    let sections: Vec<(SectionId, &[u8])> = metas
        .iter()
        .map(|m| {
            Ok((
                SectionId::try_from(m.id)?,
                &bytes[m.off as usize..(m.off + m.len) as usize],
            ))
        })
        .collect::<Result<_>>()?;

    // ─ parallel fan-out (nested joins, always two closures per join)
    let (f_res, (mat_res, rest_vecs)) = rayon::join(
        || de!(BinaryTree<F>, slice!(sections, FLeaves)),
        || {
            rayon::join(
                || {
                    rayon::join(
                        || de!(BinaryTree<Mat2x2<F>>, slice!(sections, RecombineMatrices)),
                        || de!(BinaryTree<Mat2x2<F>>, slice!(sections, DecomposeMatrices)),
                    )
                },
                || de!(Vec<RationalMap<F>>, slice!(sections, RationalMaps)),
            )
        },
    );

    let ((xnn_s_res, xnn_s_inv_res), vec6) = rayon::join(
        || {
            rayon::join(
                || de!(Vec<F>, slice!(sections, XnnS)),
                || de!(Vec<F>, slice!(sections, XnnSInv)),
            )
        },
        || {
            rayon::join(
                || {
                    rayon::join(
                        || de!(Vec<F>, slice!(sections, Z0S1)),
                        || de!(Vec<F>, slice!(sections, Z1S0)),
                    )
                },
                || {
                    rayon::join(
                        || {
                            rayon::join(
                                || de!(Vec<F>, slice!(sections, Z0InvS1)),
                                || de!(Vec<F>, slice!(sections, Z1InvS0)),
                            )
                        },
                        || {
                            rayon::join(
                                || de!(Vec<F>, slice!(sections, Z0Z0RemXnnS)),
                                || de!(Vec<F>, slice!(sections, Z1Z1RemXnnS)),
                            )
                        },
                    )
                },
            )
        },
    );

    // macro to unwrap results cleanly
    macro_rules! ok {
        ($r:expr) => {
            $r.map_err(anyhow::Error::from)?
        };
    }

    let ((z0_s1_res, z1_s0_res), ((z0_inv_s1_res, z1_inv_s0_res), (z0z0_res, z1z1_res))) = vec6;

    // ─ subtree?
    let subtree = metas
        .iter()
        .find(|m| m.id == Subtree as u8)
        .map(|m| &bytes[m.off as usize..(m.off + m.len) as usize])
        .map(|sl| read_fftree_from_slice::<F>(sl).map(|(t, _)| Box::new(t)))
        .transpose()?
        .map(Some)
        .unwrap_or(None);

    // ─ build struct
    let node = FFTree {
        f: ok!(f_res),
        recombine_matrices: ok!(mat_res.0),
        decompose_matrices: ok!(mat_res.1),
        rational_maps: ok!(rest_vecs),
        xnn_s: ok!(xnn_s_res),
        xnn_s_inv: ok!(xnn_s_inv_res),
        z0_s1: ok!(z0_s1_res),
        z1_s0: ok!(z1_s0_res),
        z0_inv_s1: ok!(z0_inv_s1_res),
        z1_inv_s0: ok!(z1_inv_s0_res),
        z0z0_rem_xnn_s: ok!(z0z0_res),
        z1z1_rem_xnn_s: ok!(z1z1_res),
        subtree,
    };

    let consumed = metas.iter().map(|m| m.len as usize).sum::<usize>() + pos;
    Ok((node, consumed))
}

// Read only the fields necessary for fftree.extend() called during proof generation
fn read_minimal_fftree_from_slice<F>(bytes: &[u8]) -> Result<(FFTree<F>, usize)>
where
    F: Field + CanonicalDeserialize + Send + Sync,
{
    use SectionId::*;

    // ─ header + table
    let hdr = NodeHeader::from_slice(&bytes[..8]);
    let mut metas = Vec::<SectionMeta>::with_capacity(hdr.section_count as usize);
    let mut pos = 8;
    for _ in 0..hdr.section_count {
        metas.push(SectionMeta::from_slice(&bytes[pos..pos + 24]));
        pos += 24;
    }
    let sections: Vec<(SectionId, &[u8])> = metas
        .iter()
        .map(|m| {
            Ok((
                SectionId::try_from(m.id)?,
                &bytes[m.off as usize..(m.off + m.len) as usize],
            ))
        })
        .collect::<Result<_>>()?;

    // ─ parallel fan-out (nested joins, always two closures per join)
    let (f_res, mat_res) = rayon::join(
        || de!(BinaryTree<F>, slice!(sections, FLeaves)),
        || {
            rayon::join(
                || de!(BinaryTree<Mat2x2<F>>, slice!(sections, RecombineMatrices)),
                || de!(BinaryTree<Mat2x2<F>>, slice!(sections, DecomposeMatrices)),
            )
        },
    );

    // macro to unwrap results cleanly
    macro_rules! ok {
        ($r:expr) => {
            $r.map_err(anyhow::Error::from)?
        };
    }

    // ─ build struct
    let node = FFTree {
        f: ok!(f_res),
        recombine_matrices: ok!(mat_res.0),
        decompose_matrices: ok!(mat_res.1),
        rational_maps: vec![],
        xnn_s: vec![],
        xnn_s_inv: vec![],
        z0_s1: vec![],
        z1_s0: vec![],
        z0_inv_s1: vec![],
        z1_inv_s0: vec![],
        z0z0_rem_xnn_s: vec![],
        z1z1_rem_xnn_s: vec![],
        subtree: None,
    };

    let consumed = metas.iter().map(|m| m.len as usize).sum::<usize>() + pos;
    Ok((node, consumed))
}

/// Reads only the fields of [`ecfft::fftree::FFTree`] that are relevant for proof generation step
/// The tree structure in path may hold the complete set of data, but we only read the portion we need
/// for proof generation. This means lower memory footpring and smaller IO time.
pub(crate) fn read_minimal_fftree_from_file<F>(path: impl AsRef<Path>) -> Result<FFTree<F>>
where
    F: Field + CanonicalDeserialize + Send + Sync,
{
    const MAGIC: &[u8; 8] = b"FFTR\0\0\0\0";
    let f = File::open(path)?;
    let mmap = unsafe { Mmap::map(&f)? };
    if &mmap[..8] != MAGIC {
        bail!("not an FFTR file");
    }
    let total = u64::from_le_bytes(mmap[8..16].try_into()?) as usize;
    let slice = &mmap[16..16 + total];
    let (tree, _) = read_minimal_fftree_from_slice::<F>(slice)?;
    Ok(tree)
}

#[cfg(test)]
mod test {

    use std::{env::temp_dir, fs::File, path::Path, time::Instant};

    use ark_ff::UniformRand;
    use ark_serialize::CanonicalDeserialize;
    use ark_std::rand::thread_rng;
    use ecfft::FFTree;
    use memmap2::Mmap;

    use crate::{
        curve::Fr,
        ec_fft::build_sect_ecfft_tree,
        tree_io::{read_minimal_fftree_from_file, write_fftree_to_file},
    };

    use super::read_fftree_from_file;

    #[test]
    #[ignore]
    fn test_tree_io() {
        let now = Instant::now();
        println!("read from old file");
        let p = Path::new("tmp_tree2nd");
        let file = File::open("srs_sect/tree2nd").unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        let tree = FFTree::<Fr>::deserialize_uncompressed(&mmap[..]).unwrap();
        let elapse = now.elapsed();
        println!("finished read from old file  {:?}", elapse.as_secs());

        let now = Instant::now();
        write_fftree_to_file(&tree, p).unwrap();
        let elapse = now.elapsed();
        println!("finished write to new file  {:?}", elapse.as_secs());

        let now = Instant::now();
        let treed: FFTree<Fr> = read_fftree_from_file(p).unwrap();
        let tl = tree.f.leaves();
        let tld = treed.f.leaves();
        let elapse = now.elapsed();
        println!("finished read from new file  {:?}", elapse.as_secs());

        assert_eq!(tl, tld);
    }

    #[test]
    fn test_verify_that_extend_works_over_minimal_tree() {
        let mut rng = thread_rng();
        let temp_file_path = temp_dir().join("tree");

        let num_constraints: usize = 1 << 6;
        let base_log_n = num_constraints.ilog2() as usize + 1;
        let tree2n: FFTree<Fr> =
            build_sect_ecfft_tree(num_constraints * 2, false, base_log_n, false).unwrap();
        let num_constrants: usize = tree2n.f.leaves().len() / 2;
        let mut evals = Vec::with_capacity(num_constrants);
        for _ in 0..num_constrants {
            evals.push(Fr::rand(&mut rng));
        }
        let evals1 = tree2n.extend(&evals, ecfft::Moiety::S1);

        write_fftree_to_file(&tree2n, temp_file_path.clone()).unwrap();
        let tree2n_min = read_minimal_fftree_from_file(&temp_file_path).unwrap();
        let evals2 = tree2n_min.extend(&evals, ecfft::Moiety::S1);

        assert_eq!(evals1, evals2);
    }
}
