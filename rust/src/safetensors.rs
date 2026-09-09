//! Minimal safetensors reader: enough to pull the checkpoint's embedding
//! tables and a voice profile. Format: 8-byte little-endian header length, a
//! JSON header naming each tensor with its dtype, shape and byte offsets, then
//! the raw tensors.

use std::collections::HashMap;
use std::fs;

use serde_json::Value;

pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

pub struct File {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
}

impl File {
    pub fn open(path: &str) -> Result<File, String> {
        let buf = fs::read(path).map_err(|e| format!("{}: {e}", path))?;
        if buf.len() < 8 {
            return Err(format!("{path}: too small to be safetensors"));
        }
        let header_len = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
        if header_len > buf.len() - 8 {
            return Err(format!("{path}: bad header length"));
        }
        let header: Value = serde_json::from_slice(&buf[8..8 + header_len])
            .map_err(|e| format!("{path}: bad header: {e}"))?;
        let obj = header.as_object().ok_or("header not an object")?;
        let mut file = File {
            tensors: HashMap::new(),
            metadata: HashMap::new(),
        };
        let base = 8 + header_len;
        for (name, spec) in obj {
            if name == "__metadata__" {
                if let Some(meta) = spec.as_object() {
                    for (k, v) in meta {
                        if let Some(s) = v.as_str() {
                            file.metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = spec["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<i64> = spec["shape"]
                .as_array()
                .map(|a| a.iter().map(|d| d.as_i64().unwrap_or(0)).collect())
                .unwrap_or_default();
            let offsets = spec["data_offsets"].as_array().ok_or("no data_offsets")?;
            // .get, not [0]/[1]: a header whose data_offsets has fewer than
            // two elements would index-panic, and this is untrusted file
            // content.
            let begin = offsets
                .first()
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("tensor {name}: bad data_offsets"))?
                as usize;
            let end = offsets
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("tensor {name}: bad data_offsets"))?
                as usize;
            // Checked against base+end, with the add guarded: the slice below
            // is indexed from the end of the header, so offsets that fit the
            // file but not the data section would slice out of range and
            // panic. A library returns Err on a corrupt checkpoint.
            let stop = base
                .checked_add(end)
                .ok_or_else(|| format!("tensor {name}: data_offsets overflow"))?;
            if begin > end || stop > buf.len() {
                return Err(format!(
                    "tensor {name}: spans {begin}..{end} of a {}-byte payload, file is \
                     truncated or the header is corrupt",
                    buf.len().saturating_sub(base)
                ));
            }
            // The shape must account for exactly the bytes claimed.
            //
            // The range check above stops a slice panic, but callers read
            // `shape` to size their work: a header declaring `[256]` over four
            // bytes of payload is not a bad tensor, it is a reader computing
            // with a length the data does not have. The accessors below also
            // use `chunks_exact`, which silently drops a partial tail; with
            // this check a partial tail cannot exist.
            let width = byte_width(&dtype)
                .ok_or_else(|| format!("tensor {name}: unknown dtype {dtype}"))?;
            let mut elements: usize = 1;
            for dim in &shape {
                // Non-negative, checked before the cast: a negative dimension
                // would wrap to an enormous usize and the overflow check below
                // would then be the only thing standing between a corrupt
                // header and an allocation the size of the address space.
                let dim = usize::try_from(*dim)
                    .map_err(|_| format!("tensor {name}: negative dimension in {shape:?}"))?;
                elements = elements
                    .checked_mul(dim)
                    .ok_or_else(|| format!("tensor {name}: shape {shape:?} overflows"))?;
            }
            let declared = elements
                .checked_mul(width)
                .ok_or_else(|| format!("tensor {name}: shape {shape:?} overflows"))?;
            if declared != end - begin {
                return Err(format!(
                    "tensor {name}: declares shape {shape:?} of {dtype} ({declared} bytes) \
                     but occupies {} bytes, the header does not describe the payload",
                    end - begin
                ));
            }
            file.tensors.insert(
                name.clone(),
                Tensor {
                    dtype,
                    shape,
                    data: buf[base + begin..stop].to_vec(),
                },
            );
        }
        Ok(file)
    }

    /// F32 data, or F16 upcast exactly.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("no tensor {name}"))?;
        match t.dtype.as_str() {
            "F32" => {
                let mut out = Vec::with_capacity(t.data.len() / 4);
                for chunk in t.data.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
                Ok(out)
            }
            "F16" => Ok(t
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half_to_f32(u16::from_le_bytes(*c)))
                .collect()),
            other => Err(format!("{name}: expected F32/F16, got {other}")),
        }
    }

    /// I64 data.
    pub fn i64(&self, name: &str) -> Result<Vec<i64>, String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("no tensor {name}"))?;
        if t.dtype != "I64" {
            return Err(format!("{name}: expected I64, got {}", t.dtype));
        }
        Ok(t.data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect())
    }
}

/// One tensor to write: its name, dtype, shape and little-endian bytes.
pub struct Entry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

/// The order the safetensors library lays tensors out in: widest dtype first,
/// then by name. Only the order the reader inverts; the format accepts any.
fn dtype_rank(dtype: &str) -> usize {
    [
        "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16", "I32", "U32", "F32",
        "F64", "I64", "U64",
    ]
    .iter()
    .position(|d| *d == dtype)
    .unwrap_or(0)
}

/// Write tensors and metadata as a safetensors file: the 8-byte header
/// length, the JSON header padded to a multiple of eight with spaces, then
/// the payloads in header order. Owner-only permissions on POSIX.
///
/// # Errors
///
/// For a shape that does not account for the bytes, and everything the file
/// write returns.
pub fn write(
    path: &std::path::Path,
    entries: &[Entry],
    metadata: &HashMap<String, String>,
) -> Result<(), String> {
    let mut sorted: Vec<&Entry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        dtype_rank(&b.dtype)
            .cmp(&dtype_rank(&a.dtype))
            .then_with(|| a.name.cmp(&b.name))
    });
    let mut fields: Vec<String> = Vec::new();
    if !metadata.is_empty() {
        let meta: std::collections::BTreeMap<&String, &String> = metadata.iter().collect();
        fields.push(format!(
            "\"__metadata__\":{}",
            serde_json::to_string(&meta).map_err(|e| e.to_string())?
        ));
    }
    let mut offset = 0usize;
    for e in &sorted {
        let width =
            byte_width(&e.dtype).ok_or_else(|| format!("{}: unknown dtype {}", e.name, e.dtype))?;
        let elements: i64 = e.shape.iter().product();
        if elements * width as i64 != e.data.len() as i64 {
            return Err(format!(
                "{}: shape {:?} of {} is {} bytes, data is {}",
                e.name,
                e.shape,
                e.dtype,
                elements * width as i64,
                e.data.len()
            ));
        }
        fields.push(format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            serde_json::to_string(&e.name).map_err(|err| err.to_string())?,
            e.dtype,
            serde_json::to_string(&e.shape).map_err(|err| err.to_string())?,
            offset,
            offset + e.data.len()
        ));
        offset += e.data.len();
    }
    let mut header = format!("{{{}}}", fields.join(","));
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for e in &sorted {
        out.extend_from_slice(&e.data);
    }
    fs::write(path, &out).map_err(|e| format!("{}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Reported, not discarded. The doc above promises owner-only
        // permissions, so a chmod that fails leaves a voice profile readable
        // by everyone with the caller told nothing.
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("{}: cannot restrict to owner-only: {e}", path.display()))?;
    }
    Ok(())
}

/// Bytes per element, or `None` for a dtype this reader does not know.
///
/// Listed rather than inferred: an unknown dtype must be refused at load, not
/// discovered later by whichever accessor is asked for it first.
fn byte_width(dtype: &str) -> Option<usize> {
    match dtype {
        "F64" | "I64" | "U64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "I8" | "U8" | "BOOL" => Some(1),
        _ => None,
    }
}

/// Exact fp16 -> fp32 upcast.
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    if exp == 0 {
        return sign * (frac as f32) * 2f32.powi(-24);
    }
    if exp == 31 {
        return if frac == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    sign * (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
}
