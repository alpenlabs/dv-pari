//! Utilities for serialising and deserialising vectors of finite‑field elements (`Fr`) and
//! elliptic‑curve points (`CurvePoint`) to and from binary files.
//!
//! ## Binary format
//! Every file produced by the `*_to_file` helpers follows the same layout:
//! 1. **Length prefix** – a little‑endian `u64` indicating how many elements are stored.
//! 2. **Payload**      – the concatenation of each element

use anyhow::Context;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use memmap2::Mmap;
use rayon::{
    iter::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use std::io::BufWriter;
use std::{
    fs::File,
    io::{Cursor, Write},
    path::Path,
};

use crate::curve::{CurvePoint, Fr};

/// Serialises a slice of [`Fr`] to binary `path`.
///
/// The function writes:
/// 1. A little‑endian `u64` length prefix.
/// 2. The uncompressed canonical representation of each element, in order.
///    Uncompressed size chosen to reduce the cost of deserialization during file read
///
/// To maximise throughput each element is first serialised **in parallel** using [`rayon`], then
/// written sequentially.  This avoids contention on the underlying [`BufWriter`].
///
/// # Arguments
/// * `path`   – Destination file path.
/// * `points` – Slice of field elements to serialise.
///
/// # Errors
/// Returns an error if the file cannot be created, any element fails to serialise, or if writing
/// to disk fails.
pub(crate) fn write_fr_vec_to_file<P: AsRef<Path>>(path: P, points: &[Fr]) -> anyhow::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    println!("write_fr_vec_to_file");
    // 1. length prefix
    let len = points.len() as u64;
    f.write_all(&len.to_le_bytes())?;

    // 2. points (compressed)
    let serialized_points: Vec<Result<Vec<u8>, anyhow::Error>> = points
        .par_iter()
        .map(|p| {
            let mut buf = Vec::with_capacity(p.uncompressed_size()); // Use uncompressed_size for clarity
            p.serialize_uncompressed(&mut buf)
                .context("Failed to serialize point")?;
            Ok(buf)
        })
        .collect();

    // 3. Write serialized points sequentially
    for result_buf in serialized_points {
        let buf = result_buf.context("A point failed to serialize in parallel")?;
        f.write_all(&buf)
            .context("Failed to write serialized point data")?;
    }

    f.flush().context("Failed to flush BufWriter")?;
    Ok(())
}

/// Serialises a slice of [`CurvePoint`]s to binary `path`.
///
/// The wire format matches [`write_fr_vec_to_file`], but uses the point's 30‑byte compressed
/// representation returned by [`CurvePoint::to_bytes`].
///
/// # Arguments
/// * `path`   – Destination file path.
/// * `points` – Slice of points to serialise.
///
/// # Errors
/// Returns an error if the file cannot be created, any point fails to serialise, or if writing to
/// disk fails.
pub(crate) fn write_point_vec_to_file<P: AsRef<Path>>(
    path: P,
    points: &[CurvePoint],
) -> anyhow::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    println!("write_point_vec_to_file");
    // 1. length prefix
    let len = points.len() as u64;
    f.write_all(&len.to_le_bytes())?;

    // 2. points (compressed)
    let serialized_points: Vec<Result<Vec<u8>, anyhow::Error>> = points
        .par_iter()
        .map(|p| {
            let buf = p.to_bytes();
            Ok(buf.to_vec())
        })
        .collect();

    // 3. Write serialized points sequentially
    for result_buf in serialized_points {
        let buf = result_buf.context("A point failed to serialize in parallel")?;
        f.write_all(&buf)
            .context("Failed to write serialized point data")?;
    }

    f.flush().context("Failed to flush BufWriter")?;
    Ok(())
}

/// Reads a vector of [`Fr`]s from binary `path` that was written by
/// [`write_fr_vec_to_file`].
///
/// The function memory‑maps the file (read‑only) and deserialises the payload **in parallel**.
///
/// # Arguments
/// * `path` – Source file path.
///
/// # Errors
/// Returns an error if the file is malformed, too short, or any element fails to deserialise.
pub(crate) fn read_fr_vec_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<Fr>> {
    let file = File::open(path).context("Failed to open file")?;
    // Safety: Creating a memory map is unsafe. Ensure file isn't truncated/modified while mapped.
    let mmap = unsafe { Mmap::map(&file)? };
    const FR_UNCOMPRESSED_SIZE: usize = 29;
    let mut current_pos = 0;

    // 1. Length prefix from mmap
    if mmap.len() < 8 {
        return Err(anyhow::anyhow!("File too short for length prefix"));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&mmap[0..8]);
    current_pos += 8;
    let len = u64::from_le_bytes(len_bytes) as usize;

    if len == 0 {
        return Ok(Vec::new());
    }

    // 2. Point data from mmap
    let expected_data_end = current_pos + (len * FR_UNCOMPRESSED_SIZE);
    if mmap.len() < expected_data_end {
        return Err(anyhow::anyhow!(
            "File too short for expected point data (compressed)"
        ));
    }
    let all_point_data_slice = &mmap[current_pos..expected_data_end];

    // 3. Parallelize deserialization from the mmap slice
    let points_results: Vec<Result<Fr, anyhow::Error>> = all_point_data_slice
        .par_chunks_exact(FR_UNCOMPRESSED_SIZE)
        .map(|chunk| {
            let mut cursor = Cursor::new(chunk);
            Fr::deserialize_uncompressed_unchecked(&mut cursor).map_err(|e| {
                anyhow::Error::from(e).context("Failed to deserialize point (compressed) from mmap")
            })
        })
        .collect();

    let points = points_results.into_iter().collect::<Result<Vec<_>, _>>()?;

    if points.len() != len {
        return Err(anyhow::anyhow!(
        "Deserialization (mmap, compressed) resulted in incorrect number of points. Expected {}, got {}.",
        len,
        points.len()
    ));
    }

    Ok(points)
}

/// Reads a vector of [`CurvePoint`]s from binary `path` that was written by
/// [`write_point_vec_to_file`].
///
/// The function memory‑maps the file and deserialises each fixed‑width 30‑byte chunk **in
/// parallel**.
///
/// # Arguments
/// * `path` – Source file path.
///
/// # Errors
/// Returns an error if the file is malformed, too short, or any point fails to deserialise.
pub(crate) fn read_point_vec_from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<CurvePoint>> {
    let file = File::open(path).context("Failed to open file")?;
    // Safety: Creating a memory map is unsafe. Ensure file isn't truncated/modified while mapped.
    let mmap = unsafe { Mmap::map(&file)? };
    const PT_COMPRESSED_SIZE: usize = 30;
    let mut current_pos = 0;

    // 1. Length prefix from mmap
    if mmap.len() < 8 {
        return Err(anyhow::anyhow!("File too short for length prefix"));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&mmap[0..8]);
    current_pos += 8;
    let len = u64::from_le_bytes(len_bytes) as usize;

    if len == 0 {
        return Ok(Vec::new());
    }

    // 2. Point data from mmap
    let expected_data_end = current_pos + (len * PT_COMPRESSED_SIZE);
    if mmap.len() < expected_data_end {
        return Err(anyhow::anyhow!(
            "File too short for expected point data (compressed)"
        ));
    }
    let all_point_data_slice = &mmap[current_pos..expected_data_end];

    // 3. Parallelize deserialization from the mmap slice
    let points_results: Vec<Result<CurvePoint, anyhow::Error>> = all_point_data_slice
        .par_chunks_exact(PT_COMPRESSED_SIZE)
        .map(|chunk| {
            let cursor = Cursor::new(chunk);
            let mut x: [u8; 30] = cursor.into_inner().try_into().unwrap();
            let pt = CurvePoint::from_bytes(&mut x);
            Ok(pt)
        })
        .collect();

    let points = points_results.into_iter().collect::<Result<Vec<_>, _>>()?;

    if points.len() != len {
        return Err(anyhow::anyhow!(
        "Deserialization (mmap, compressed) resulted in incorrect number of points. Expected {}, got {}.",
        len,
        points.len()
    ));
    }

    Ok(points)
}

#[cfg(test)]
mod test {
    use std::env::temp_dir;

    use xs233_sys::{xsk233_generator, xsk233_neutral};

    use crate::curve::CurvePoint;

    use super::{read_point_vec_from_file, write_point_vec_to_file};

    // Verifies that IO operations yield the original elements
    #[test]
    fn test_points_match_after_write_and_read() {
        unsafe {
            let pts = vec![
                CurvePoint(xsk233_generator),
                CurvePoint(xsk233_neutral),
                CurvePoint(xsk233_generator),
                CurvePoint(xsk233_neutral),
            ];
            let mut path = temp_dir();
            path.push("test_test_read_write_points");
            write_point_vec_to_file(path.clone(), &pts).unwrap();
            let pts2 = read_point_vec_from_file(path).unwrap();
            assert_eq!(pts, pts2);
        }
    }
}
