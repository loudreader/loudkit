//! The record beside a set of exported graphs, and the check that holds them
//! to it.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

/// What the exporter writes beside the graphs it wrote.
pub const EXPORT_RECORD: &str = "export.json";

/// The tool that writes it, and what a refusal tells the reader to run.
const EXPORT_TOOL: &str = "tools/export_onnx.py";

/// The fields that make one export's identity, in the order a message prints
/// them. The first three are the engine the graphs were traced against; the
/// fourth is the renderer inside it.
const RECORD_KEYS: [&str; 4] = [
    "checkpoint_sha256",
    "algorithm_fingerprint",
    "euler_steps",
    "estimator_sha256",
];

/// One member's record, rendered for comparison. Each field holds the value
/// the way Python's `repr()` prints it, so a set that agrees here agrees there
/// and a refusal reads the same in both.
type Identity = [String; RECORD_KEYS.len()];

/// Refuse a graph set whose members did not come from one export of this
/// checkpoint.
///
/// Graphs in one folder look like a set and need not be one: the exporters
/// take a stage list, so a run that names the renderer leaves the other graphs
/// as they were, and a mixed folder speaks one checkpoint's tokens through
/// another checkpoint's renderer. [`EXPORT_RECORD`] beside the set records,
/// per member, the checkpoint digest, the fingerprint, the step count and the
/// estimator digest; every member must agree with the others and with the
/// checkpoint being loaded.
///
/// An absent record is a warning rather than a refusal, because sets exported
/// before the record exist and stranding them buys nothing a sentence cannot
/// say.
pub(crate) fn check_export_record(
    assets: &Path,
    ckpt_path: &Path,
    manifest: &HashMap<String, Value>,
    fingerprint: &str,
    euler_steps: usize,
    graphs: &[&str],
) -> Result<(), String> {
    let path = assets.join(EXPORT_RECORD);
    if !path.is_file() {
        eprintln!(
            "loudkit: warning: {} carries no {EXPORT_RECORD}, so nothing says its {} graphs \
             came from one export of one checkpoint. Re-export with {EXPORT_TOOL} to record it.",
            assets.display(),
            graphs.len()
        );
        return Ok(());
    }
    let seen = recorded_graphs(&path)?;
    for name in graphs {
        if !seen.contains_key(*name) {
            return Err(format!(
                "{} does not record {name}, so it came from some other run than the ones \
                 it does record. Re-export the set",
                path.display()
            ));
        }
    }
    let mut agreed: Option<&Identity> = None;
    for got in seen.values() {
        match agreed {
            Some(first) if first != got => {
                return Err(format!(
                    "{} is a mixed graph set: its members were exported from different \
                     inputs:\n{}\nRe-export every stage together",
                    assets.display(),
                    rows(&seen)
                ));
            }
            _ => agreed = Some(got),
        }
    }
    // Every graph asked for is recorded by now, so a set with graphs has an
    // identity; a caller asking for none has nothing to disagree with.
    let Some(agreed) = agreed else { return Ok(()) };
    let want = [
        quote(&crate::hub::file_sha256(ckpt_path)?),
        quote(fingerprint),
        euler_steps.to_string(),
    ];
    if agreed[..3] != want {
        return Err(format!(
            "{} was exported from a different engine than the one loading it:\n  graphs: {}\n  \
             checkpoint: {}\nRe-export against {}",
            assets.display(),
            engine_fields(&agreed[..3]),
            engine_fields(&want),
            file_name(ckpt_path)
        ));
    }
    // A set traced from a swapped-in estimator agrees with itself and describes
    // a renderer the checkpoint does not; compared only when both sides record
    // one.
    let packed = packed_estimator(manifest);
    match packed {
        Some(packed) if agreed[3] != "None" && agreed[3] != quote(&packed) => Err(format!(
            "{} was traced with an estimator the checkpoint was not packed from:\n  traced:  \
             {}\n  packed:  {packed}\nThat is a different renderer than {} describes, and its \
             fingerprint does not cover the difference. Re-export without --estimator-ckpt, or \
             pack the estimator you traced",
            assets.display(),
            agreed[3].trim_matches('\''),
            file_name(ckpt_path)
        )),
        _ => Ok(()),
    }
}

/// Read the record's graph block. An absent block is an unreadable record; a
/// block that is not a mapping of name to its fields is named as itself,
/// because the remedy differs.
fn recorded_graphs(path: &Path) -> Result<HashMap<String, Identity>, String> {
    let unreadable =
        |what: String| format!("{}: unreadable export record ({what})", path.display());
    let body = std::fs::read_to_string(path).map_err(|e| unreadable(e.to_string()))?;
    let root: Value = serde_json::from_str(&body).map_err(|e| unreadable(e.to_string()))?;
    let block = root
        .get("graphs")
        .ok_or_else(|| unreadable("no 'graphs' block".to_string()))?;
    let malformed = format!(
        "{}: the 'graphs' block is not a mapping of name to its export record. \
         Re-export the set with {EXPORT_TOOL}",
        path.display()
    );
    let entries = block.as_object().ok_or_else(|| malformed.clone())?;
    let mut seen = HashMap::with_capacity(entries.len());
    for (name, entry) in entries {
        seen.insert(
            name.clone(),
            as_identity(entry).ok_or_else(|| malformed.clone())?,
        );
    }
    Ok(seen)
}

/// Read one member's entry, declining a value shape the exporter never writes.
/// An absent field and a null one are the same answer, so a record written
/// before a field existed still compares.
fn as_identity(entry: &Value) -> Option<Identity> {
    let entry = entry.as_object()?;
    let mut out: Identity = Default::default();
    for (slot, key) in out.iter_mut().zip(RECORD_KEYS) {
        *slot = match entry.get(key) {
            None | Some(Value::Null) => "None".to_string(),
            Some(Value::String(text)) => quote(text),
            Some(Value::Number(number)) if number.is_i64() || number.is_u64() => number.to_string(),
            _ => return None,
        };
    }
    Some(out)
}

/// The sha256 of the estimator the checkpoint was packed from, if its sources
/// say.
fn packed_estimator(manifest: &HashMap<String, Value>) -> Option<String> {
    manifest
        .get("sources")?
        .as_object()?
        .values()
        .find(|entry| entry.get("role").and_then(Value::as_str) == Some("estimator"))?
        .get("sha256")?
        .as_str()
        .map(str::to_string)
}

/// A value the way Python's `repr()` prints a string, so a refusal reads the
/// same in both.
fn quote(value: &str) -> String {
    format!("'{value}'")
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// The three fields the engine is compared on.
fn engine_fields(got: &[String]) -> String {
    let parts: Vec<String> = got
        .iter()
        .zip(RECORD_KEYS)
        .map(|(value, key)| format!("{}: {value}", quote(key)))
        .collect();
    format!("{{{}}}", parts.join(", "))
}

/// Every member's record, one per line, for a set that disagrees with itself.
fn rows(seen: &HashMap<String, Identity>) -> String {
    let mut names: Vec<&String> = seen.keys().collect();
    names.sort();
    names
        .iter()
        .map(|name| {
            let fields: Vec<String> = seen[*name]
                .iter()
                .zip(RECORD_KEYS)
                .map(|(value, key)| format!("{}: {value}", quote(key)))
                .collect();
            format!("  {name}: {{{}}}", fields.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GRAPHS: [&str; 6] = [
        "t3_cond.onnx",
        "t3_prefill.onnx",
        "t3_step.onnx",
        "flow_encoder.onnx",
        "flow_estimator.onnx",
        "vocoder.onnx",
    ];

    /// A scratch directory holding a stand-in for the checkpoint, whose digest
    /// the record has to name.
    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("loudkit-export-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ckpt.safetensors"), b"ckpt").unwrap();
        dir
    }

    /// The fields a record has to carry to match the checkpoint in `dir`.
    fn match_fields(dir: &Path) -> Value {
        json!({
            "checkpoint_sha256": crate::hub::file_sha256(&dir.join("ckpt.safetensors")).unwrap(),
            "algorithm_fingerprint": "ae115ff943456910",
            "euler_steps": 2,
        })
    }

    /// Write a record whose members all carry `fields`, and `odd` for the
    /// vocoder when one is given.
    fn write_record(dir: &Path, fields: &Value, odd: Option<&Value>) {
        let mut graphs = serde_json::Map::new();
        for name in GRAPHS {
            graphs.insert(name.to_string(), fields.clone());
        }
        if let Some(odd) = odd {
            graphs.insert("vocoder.onnx".to_string(), odd.clone());
        }
        let body = json!({"format": "loudkit-onnx-export", "graphs": graphs});
        std::fs::write(dir.join(EXPORT_RECORD), body.to_string()).unwrap();
    }

    fn check_with(dir: &Path, manifest: &HashMap<String, Value>) -> Result<(), String> {
        check_export_record(
            dir,
            &dir.join("ckpt.safetensors"),
            manifest,
            "ae115ff943456910",
            2,
            &GRAPHS,
        )
    }

    fn check(dir: &Path) -> Result<(), String> {
        check_with(dir, &HashMap::new())
    }

    #[test]
    fn a_matching_record_loads_and_no_record_loads() {
        let dir = scratch("match");
        let fields = match_fields(&dir);
        write_record(&dir, &fields, None);
        assert_eq!(check(&dir), Ok(()));

        std::fs::remove_file(dir.join(EXPORT_RECORD)).unwrap();
        assert_eq!(
            check(&dir),
            Ok(()),
            "a set exported before the record must load"
        );
    }

    #[test]
    fn a_record_naming_another_engine_is_refused() {
        let dir = scratch("engine");
        let mut fields = match_fields(&dir);
        fields["algorithm_fingerprint"] = json!("5cfefec451bcedd1");
        write_record(&dir, &fields, None);
        let err = check(&dir).unwrap_err();
        assert!(
            err.contains("was exported from a different engine than the one loading it"),
            "{err}"
        );
        assert!(
            err.contains("'algorithm_fingerprint': '5cfefec451bcedd1'"),
            "{err}"
        );
        assert!(
            err.contains("'algorithm_fingerprint': 'ae115ff943456910'"),
            "{err}"
        );
    }

    #[test]
    fn one_stage_re_exported_on_its_own_is_a_mixed_set() {
        let dir = scratch("mixed");
        let fields = match_fields(&dir);
        let mut odd = fields.clone();
        odd["euler_steps"] = json!(7);
        write_record(&dir, &fields, Some(&odd));
        let err = check(&dir).unwrap_err();
        assert!(err.contains("is a mixed graph set"), "{err}");
        assert!(err.contains("vocoder.onnx: {"), "{err}");
    }

    #[test]
    fn a_member_the_record_does_not_name_is_refused() {
        let dir = scratch("short");
        std::fs::write(
            dir.join(EXPORT_RECORD),
            json!({"format": "loudkit-onnx-export", "graphs": {}}).to_string(),
        )
        .unwrap();
        let err = check(&dir).unwrap_err();
        assert!(err.contains("does not record t3_cond.onnx"), "{err}");
    }

    #[test]
    fn a_record_that_is_not_one_names_itself() {
        let dir = scratch("junk");
        for (body, want) in [
            (
                r#"{"graphs": [1, 2]}"#,
                "is not a mapping of name to its export record",
            ),
            (
                r#"{"graphs": {"t3_cond.onnx": {"euler_steps": [2]}}}"#,
                "is not a mapping",
            ),
            (
                r#"{"format": "loudkit-onnx-export"}"#,
                "unreadable export record",
            ),
            ("{", "unreadable export record"),
        ] {
            std::fs::write(dir.join(EXPORT_RECORD), body).unwrap();
            let err = check(&dir).unwrap_err();
            assert!(err.contains(want), "{body}: {err}");
        }
    }

    /// A field the record was written before, and one it carries as null, are
    /// the same answer.
    #[test]
    fn an_absent_field_and_a_null_one_compare_equal() {
        let dir = scratch("null");
        let fields = match_fields(&dir);
        let mut null = fields.clone();
        null["estimator_sha256"] = Value::Null;
        write_record(&dir, &fields, Some(&null));
        assert_eq!(check(&dir), Ok(()));
    }

    /// A renderer traced from a swapped-in estimator agrees with itself and
    /// describes a checkpoint this one is not.
    #[test]
    fn a_swapped_estimator_is_refused() {
        let dir = scratch("estimator");
        let manifest: HashMap<String, Value> = HashMap::from([(
            "sources".to_string(),
            json!({"flow.pt": {"role": "estimator", "sha256": "a".repeat(64)}}),
        )]);
        let mut fields = match_fields(&dir);
        fields["estimator_sha256"] = json!("b".repeat(64));
        write_record(&dir, &fields, None);
        let err = check_with(&dir, &manifest).unwrap_err();
        assert!(
            err.contains("traced with an estimator the checkpoint was not packed from"),
            "{err}"
        );
        // The estimator the checkpoint was packed from is the one it accepts.
        fields["estimator_sha256"] = json!("a".repeat(64));
        write_record(&dir, &fields, None);
        assert_eq!(check_with(&dir, &manifest), Ok(()));
    }
}
