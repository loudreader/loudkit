use loudkit::voice;

/// The asset-backed half declines when the profile is not there.
///
/// Every other asset-backed suite in this directory reads
/// `LOUDKIT_REQUIRE_ASSETS` before failing, so a checkout without the
/// enrollment fixture reports a skip rather than a failure that says nothing
/// about the crate: rust-34.
fn assets_required() -> bool {
    std::env::var("LOUDKIT_REQUIRE_ASSETS").is_ok_and(|v| !v.is_empty() && v != "0")
}

#[test]
fn legacy_python_profile_and_pause_enrolment_roundtrip() {
    let source = format!(
        "{}/../tests/data/enrollment/profile.safetensors",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&source).is_file() {
        assert!(
            !assets_required(),
            "LOUDKIT_REQUIRE_ASSETS is set and {source} is missing"
        );
        return;
    }
    let mut profile = voice::load(&source).unwrap();
    assert_eq!(profile.enrolment, "first-10s");
    for strategy in ["first-10s", "first-10s-pause"] {
        profile.enrolment = strategy.into();
        let path = std::env::temp_dir().join(format!(
            "loudkit-voice-roundtrip-{}.safetensors",
            std::process::id()
        ));
        profile.save(&path).unwrap();
        let out = voice::load(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(out.enrolment, strategy);
        assert_eq!(out.speaker_embedding, profile.speaker_embedding);
        assert_eq!(out.flow_embedding, profile.flow_embedding);
        assert_eq!(out.prompt_tokens, profile.prompt_tokens);
        assert_eq!(out.prompt_mel, profile.prompt_mel);
        assert_eq!(out.cond_prompt_tokens, profile.cond_prompt_tokens);
    }
}
