use std::{
    collections::{HashMap, HashSet},
    io::{Cursor, Read},
};

use bytemuck::{try_cast_slice, Pod};
use glam::{Vec3, Vec4};
use half::f16;
use zip::ZipArchive;

use super::{line::LineFile, line::TrackScalarFile, File};

/// Parse a `.trx` container archive into discrete `LineFile` bundles and `TrackScalarFile`s.
pub fn load(file: File) -> (Vec<LineFile>, Vec<TrackScalarFile>) {
    let Ok(mut archive) = ZipArchive::new(Cursor::new(file.data())) else {
        log::error!("Failed to open TRX archive: {}", file.name);
        return (Vec::new(), Vec::new());
    };

    let base_name = file.name.replace(".trx", "");
    let mut pos_data: Option<(String, Vec<u8>)> = None;
    let mut off_data: Option<(String, Vec<u8>)> = None;
    let mut dpv: HashMap<String, Vec<f32>> = HashMap::new();
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

    for i in 0..archive.len() {
        let Ok(mut entry) = archive.by_index(i) else { continue };
        if entry.is_dir() { continue; }

        let path = entry.name().to_string();
        let file_name = path.rsplit('/').next().unwrap_or(&path).to_string();

        if file_name.starts_with("positions.3.") {
            let mut data = Vec::new();
            if entry.read_to_end(&mut data).is_ok() {
                pos_data = Some((file_name, data));
            }
        } else if file_name.starts_with("offsets.") {
            let mut data = Vec::new();
            if entry.read_to_end(&mut data).is_ok() {
                off_data = Some((file_name, data));
            }
        } else if path.starts_with("dpv/") {
            let mut data = Vec::new();
            if entry.read_to_end(&mut data).is_ok() {
                let ext = file_name.rsplit('.').next().unwrap_or("");
                let name = file_name.split('.').next().unwrap_or(&file_name).to_string();
                dpv.insert(name, decode_floats(&data, ext));
            }
        } else if path.starts_with("groups/") {
            let mut data = Vec::new();
            if entry.read_to_end(&mut data).is_ok() {
                let ext = file_name.rsplit('.').next().unwrap_or("");
                let name = file_name.split('.').next().unwrap_or(&file_name).to_string();
                groups.insert(name, decode_indices(&data, ext));
            }
        }
    }

    let Some((pos_name, pos_bytes)) = pos_data else {
        log::error!("TRX archive `{}` missing positions array", file.name);
        return (Vec::new(), Vec::new());
    };

    let pos_ext = pos_name.rsplit('.').next().unwrap_or("");
    let points = decode_positions(&pos_bytes, pos_ext);

    let offsets = if let Some((off_name, off_bytes)) = off_data {
        let off_ext = off_name.rsplit('.').next().unwrap_or("");
        decode_offsets(&off_bytes, off_ext)
    } else {
        vec![0, points.len()]
    };

    let active_scalar = dpv
        .remove("fa")
        .map(|v| ("fa".to_string(), v))
        .or_else(|| dpv.into_iter().find(|(k, _)| !k.starts_with("color")));

    // Helper closure to assemble streamline lines directly from points slice without cloning
    let build_lines = |indices: &[usize]| -> Vec<Vec<Vec4>> {
        let mut bundle_lines = Vec::with_capacity(indices.len());
        for &idx in indices {
            let start = offsets[idx];
            let end = offsets.get(idx + 1).copied().unwrap_or(points.len()).min(points.len());
            if end > start && end - start >= 2 {
                bundle_lines.push(
                    points[start..end]
                        .iter()
                        .map(|p| Vec4::new(p.x, p.y, p.z, 1.0))
                        .collect(),
                );
            }
        }
        bundle_lines
    };

    let mut line_files: Vec<LineFile> = Vec::new();
    let mut scalar_files: Vec<TrackScalarFile> = Vec::new();

    if groups.is_empty() {
        let all_indices: Vec<usize> = (0..offsets.len().saturating_sub(1)).collect();
        let lines = build_lines(&all_indices);

        if let Some((s_name, s_vals)) = active_scalar {
            scalar_files.push(TrackScalarFile::new(
                &format!("{base_name}.{s_name}"),
                normalize_scalars(&s_vals),
            ));
        }

        line_files.push(LineFile::new(base_name, lines));
    } else {
        let mut assigned: HashSet<usize> = HashSet::new();

        for (group_name, indices) in groups {
            for &idx in &indices {
                assigned.insert(idx);
            }

            let group_lines = build_lines(&indices);
            if group_lines.is_empty() {
                continue;
            }

            let bundle_name = format!("{base_name}_{group_name}");

            if let Some((ref s_name, ref s_vals)) = active_scalar {
                let mut group_scalars = Vec::new();
                for &idx in &indices {
                    let start = offsets[idx];
                    let end = offsets.get(idx + 1).copied().unwrap_or(points.len()).min(points.len());
                    if end <= s_vals.len() {
                        group_scalars.extend_from_slice(&s_vals[start..end]);
                    }
                }
                scalar_files.push(TrackScalarFile::new(
                    &format!("{bundle_name}.{s_name}"),
                    normalize_scalars(&group_scalars),
                ));
            }

            line_files.push(LineFile::new(bundle_name, group_lines));
        }

        let total_streamlines = offsets.len().saturating_sub(1);
        let unassigned_indices: Vec<usize> = (0..total_streamlines)
            .filter(|i| !assigned.contains(i))
            .collect();

        if !unassigned_indices.is_empty() {
            let unassigned_lines = build_lines(&unassigned_indices);
            let bundle_name = format!("{base_name}_unassigned");

            if let Some((ref s_name, ref s_vals)) = active_scalar {
                let mut unassigned_scalars = Vec::new();
                for &idx in &unassigned_indices {
                    let start = offsets[idx];
                    let end = offsets.get(idx + 1).copied().unwrap_or(points.len()).min(points.len());
                    if end <= s_vals.len() {
                        unassigned_scalars.extend_from_slice(&s_vals[start..end]);
                    }
                }
                scalar_files.push(TrackScalarFile::new(
                    &format!("{bundle_name}.{s_name}"),
                    normalize_scalars(&unassigned_scalars),
                ));
            }

            line_files.push(LineFile::new(bundle_name, unassigned_lines));
        }
    }

    (line_files, scalar_files)
}

fn cast_bytes<T: Pod + Default>(bytes: &[u8]) -> Vec<T> {
    try_cast_slice(bytes)
        .map(|s| s.to_vec())
        .unwrap_or_else(|_| {
            let count = bytes.len() / std::mem::size_of::<T>();
            let mut out = vec![T::default(); count];
            bytemuck::cast_slice_mut(&mut out).copy_from_slice(&bytes[..count * std::mem::size_of::<T>()]);
            out
        })
}

fn decode_positions(bytes: &[u8], ext: &str) -> Vec<Vec3> {
    match ext {
        "float16" | "f16" => cast_bytes::<u16>(bytes)
            .chunks_exact(3)
            .map(|c| Vec3::new(
                f16::from_bits(c[0]).to_f32(),
                f16::from_bits(c[1]).to_f32(),
                f16::from_bits(c[2]).to_f32(),
            ))
            .collect(),
        "float64" | "f64" => cast_bytes::<f64>(bytes)
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
            .collect(),
        _ => cast_bytes::<f32>(bytes)
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect(),
    }
}

fn decode_offsets(bytes: &[u8], ext: &str) -> Vec<usize> {
    match ext {
        "uint64" | "u64" | "int64" | "i64" => {
            cast_bytes::<u64>(bytes).iter().map(|&x| x as usize).collect()
        }
        _ => cast_bytes::<u32>(bytes).iter().map(|&x| x as usize).collect(),
    }
}

fn decode_indices(bytes: &[u8], ext: &str) -> Vec<usize> {
    decode_offsets(bytes, ext)
}

fn decode_floats(bytes: &[u8], ext: &str) -> Vec<f32> {
    match ext {
        "float16" | "f16" => cast_bytes::<u16>(bytes).iter().map(|&b| f16::from_bits(b).to_f32()).collect(),
        "float64" | "f64" => cast_bytes::<f64>(bytes).iter().map(|&v| v as f32).collect(),
        "uint8" | "u8" => bytes.iter().map(|&b| b as f32).collect(),
        "int8" | "i8" => cast_bytes::<i8>(bytes).iter().map(|&b| b as f32).collect(),
        "uint16" | "u16" => cast_bytes::<u16>(bytes).iter().map(|&v| v as f32).collect(),
        "int16" | "i16" => cast_bytes::<i16>(bytes).iter().map(|&v| v as f32).collect(),
        "uint32" | "u32" => cast_bytes::<u32>(bytes).iter().map(|&v| v as f32).collect(),
        "int32" | "i32" => cast_bytes::<i32>(bytes).iter().map(|&v| v as f32).collect(),
        _ => cast_bytes::<f32>(bytes),
    }
}

fn normalize_scalars(values: &[f32]) -> Vec<f32> {
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if max > min && (min < 0.0 || max > 1.0) {
        let range = max - min;
        values.iter().map(|&v| (v - min) / range).collect()
    } else {
        values.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    #[test]
    fn test_load_trx_archive() {
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buffer);
            let options = SimpleFileOptions::default();

            zip.start_file("positions.3.float16", options).unwrap();
            let pts_f32 = [
                1.0f32, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0,
                13.0, 14.0, 15.0,
            ];
            let pts_f16: Vec<u8> = pts_f32
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect();
            zip.write_all(&pts_f16).unwrap();

            zip.start_file("offsets.uint32", options).unwrap();
            let offsets: [u32; 3] = [0, 3, 5];
            zip.write_all(bytemuck::cast_slice(&offsets)).unwrap();

            zip.start_file("dpv/fa.float32", options).unwrap();
            let fa: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
            zip.write_all(bytemuck::cast_slice(&fa)).unwrap();

            zip.start_file("groups/bundle_0.uint32", options).unwrap();
            let group_members: [u32; 1] = [0];
            zip.write_all(bytemuck::cast_slice(&group_members)).unwrap();

            zip.finish().unwrap();
        }

        let file = File::new("test_tract.trx", buffer.into_inner());
        let (lines, scalars) = load(file);

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].name(), "test_tract_bundle_0");
        assert_eq!(lines[0].lines().len(), 1);
        assert_eq!(lines[0].lines()[0].len(), 3);

        assert_eq!(lines[1].name(), "test_tract_unassigned");
        assert_eq!(lines[1].lines().len(), 1);
        assert_eq!(lines[1].lines()[0].len(), 2);

        assert_eq!(scalars.len(), 2);
        assert_eq!(scalars[0].name(), "test_tract_bundle_0.fa");
        assert_eq!(scalars[0].values().len(), 3);
    }

    #[test]
    fn test_load_benchmark_dataset() {
        let path = std::path::Path::new("data-set/f16_ui64_w_metadata_small.trx");
        if path.exists() {
            let data = std::fs::read(path).unwrap();
            let file = File::new("f16_ui64_w_metadata_small.trx", data);
            let (lines, scalars) = load(file);

            assert_eq!(lines.len(), 8);
            assert!(lines.iter().any(|l| l.name().ends_with("_AF_L")));
            assert!(lines.iter().any(|l| l.name().ends_with("_unassigned")));

            assert_eq!(scalars.len(), 8);
            assert!(scalars.iter().any(|s| s.name().ends_with("_AF_L.fa")));
        }
    }
}
