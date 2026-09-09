package voice

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/loudreader/loudkit/go/safetensors"
)

// A saved profile reads back as itself, and its header carries the keys
// python/loudkit/voice.py writes.
func TestSaveThenLoadIsTheSameVoice(t *testing.T) {
	speaker := make([]float32, speakerDim)
	flow := make([]float32, flowDim)
	for i := range speaker {
		speaker[i] = float32(i%7) * 0.125
	}
	for i := range flow {
		flow[i] = float32(i%5)*0.25 - 0.5
	}
	mel := make([]float32, melBins*3)
	for i := range mel {
		mel[i] = float32(i) * 0.01
	}
	in := &Profile{
		Name:             "mine",
		Enrolment:        "first-10s",
		SpeakerEmbedding: speaker,
		FlowEmbedding:    flow,
		PromptTokens:     []int64{1, 2, 3},
		PromptMel:        mel,
		CondPromptTokens: []int64{4, 5},
		SourceSampleRate: 16000,
		Language:         "pl",
	}
	path := filepath.Join(t.TempDir(), "mine.safetensors")
	if err := in.Save(path); err != nil {
		t.Fatal(err)
	}
	out, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("round trip changed the profile:\n in %+v\nout %+v", in, out)
	}
}

// An over-long name is truncated to MaxNameChars, not refused. Python
// does the same in voice.py, and it slices by code point, so the cut has to
// land between runes rather than inside one.
func TestLoadTruncatesAnOverLongName(t *testing.T) {
	in := namedProfile(strings.Repeat("ż", MaxNameChars+50))
	path := filepath.Join(t.TempDir(), "long.safetensors")
	if err := in.Save(path); err != nil {
		t.Fatal(err)
	}
	out, err := Load(path)
	if err != nil {
		t.Fatalf("an over-long name must load, not be refused: %v", err)
	}
	if got := utf8.RuneCountInString(out.Name); got != MaxNameChars {
		t.Fatalf("name kept %d runes, want %d", got, MaxNameChars)
	}
	if out.Name != strings.Repeat("ż", MaxNameChars) {
		t.Fatalf("the cut landed inside a rune: %q", out.Name)
	}
}

// A name at the cap is carried through untouched.
func TestLoadKeepsANameAtTheCap(t *testing.T) {
	name := strings.Repeat("a", MaxNameChars)
	in := namedProfile(name)
	path := filepath.Join(t.TempDir(), "cap.safetensors")
	if err := in.Save(path); err != nil {
		t.Fatal(err)
	}
	out, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if out.Name != name {
		t.Fatalf("a name at the cap was changed: %q", out.Name)
	}
}

// namedProfile is the smallest profile Load accepts, under the given name. The
// embeddings carry a norm because Load refuses a zero speaker vector.
func namedProfile(name string) *Profile {
	speaker := make([]float32, speakerDim)
	flow := make([]float32, flowDim)
	speaker[0], flow[0] = 1, 1
	return &Profile{
		Name:             name,
		Enrolment:        "first-10s",
		SpeakerEmbedding: speaker,
		FlowEmbedding:    flow,
		PromptTokens:     []int64{1},
		PromptMel:        make([]float32, melBins),
		CondPromptTokens: []int64{2},
		SourceSampleRate: 24000,
		Language:         "pl",
	}
}

func TestSaveRefusesAMisshapenMel(t *testing.T) {
	p := &Profile{PromptMel: make([]float32, melBins+1)}
	if err := p.Save(filepath.Join(t.TempDir(), "x.safetensors")); err == nil {
		t.Error("a mel that is not (80, frames) must not be written")
	}
}

func TestLegacyAndPauseEnrollmentRoundtrip(t *testing.T) {
	p, err := Load("../../tests/data/enrollment/profile.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if p.Enrolment != "first-10s" {
		t.Fatalf("legacy strategy: %q", p.Enrolment)
	}
	p.Enrolment = "first-10s-pause"
	path := filepath.Join(t.TempDir(), "pause.safetensors")
	if err := p.Save(path); err != nil {
		t.Fatal(err)
	}
	out, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p, out) {
		t.Fatal("voice roundtrip changed profile")
	}
}

// A strategy this build does not implement is refused, not read as the
// default.
//
// The two strategies cut the prompt from different audio, so a profile made by
// a build with a third would load here, speak, and speak in a voice this build
// never enrolled, under the name the caller asked for. Nothing downstream
// reads the label, which is why the refusal has to be at the door.
// VoiceProfile._validate_enrolment refuses it with this sentence.
func TestLoadRefusesAnEnrolmentStrategyThisBuildDoesNotImplement(t *testing.T) {
	for _, strategy := range []string{"first-30s", "pause-cut", "FIRST-10S", " first-10s"} {
		p := namedProfile("mine")
		p.Enrolment = strategy
		path := filepath.Join(t.TempDir(), "unknown.safetensors")
		if err := p.Save(path); err != nil {
			t.Fatal(err)
		}
		err := errorFromLoad(t, path)
		if err == nil {
			t.Errorf("enrolment %q was accepted", strategy)
			continue
		}
		for _, want := range []string{strategy, firstWindowEnrolment, pauseCutEnrolment} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("refusal for %q does not name %q: %v", strategy, want, err)
			}
		}
	}
	// Both strategies this build does implement still load, and so does a
	// header that names none: every voice enrolled before the field existed
	// was made the first way.
	for _, strategy := range []string{firstWindowEnrolment, pauseCutEnrolment, ""} {
		p := namedProfile("mine")
		p.Enrolment = strategy
		path := filepath.Join(t.TempDir(), "known.safetensors")
		if err := p.Save(path); err != nil {
			t.Fatal(err)
		}
		out, err := Load(path)
		if err != nil {
			t.Fatalf("enrolment %q was refused: %v", strategy, err)
		}
		want := strategy
		if want == "" {
			want = firstWindowEnrolment
		}
		if out.Enrolment != want {
			t.Errorf("enrolment %q loaded as %q", strategy, out.Enrolment)
		}
	}
}

// A rate is the divisor of every duration derived from the recording, so a
// zero divides by zero and a negative reports negative seconds. Neither is
// caught by any shape. VoiceProfile._validate_values refuses it with this
// sentence.
func TestLoadRefusesANonPositiveSourceSampleRate(t *testing.T) {
	for _, rate := range []int{0, -1, -24000} {
		p := namedProfile("mine")
		p.SourceSampleRate = rate
		path := filepath.Join(t.TempDir(), "rate.safetensors")
		if err := p.Save(path); err != nil {
			t.Fatal(err)
		}
		err := errorFromLoad(t, path)
		if err == nil {
			t.Errorf("source_sample_rate %d was accepted", rate)
			continue
		}
		if !strings.Contains(err.Error(), "source_sample_rate must be positive") {
			t.Errorf("refusal for %d does not say why: %v", rate, err)
		}
	}
	// A rate the recording could have had still loads, and so does a header
	// that names none.
	for _, rate := range []int{1, 16000, 24000, 48000} {
		p := namedProfile("mine")
		p.SourceSampleRate = rate
		path := filepath.Join(t.TempDir(), "ok.safetensors")
		if err := p.Save(path); err != nil {
			t.Fatal(err)
		}
		out, err := Load(path)
		if err != nil {
			t.Fatalf("source_sample_rate %d was refused: %v", rate, err)
		}
		if out.SourceSampleRate != rate {
			t.Errorf("source_sample_rate %d loaded as %d", rate, out.SourceSampleRate)
		}
	}
}

// errorFromLoad is Load's error for a file that must not open, so a test reads
// as the refusal it is checking rather than as a nil dance.
func errorFromLoad(t *testing.T, path string) error {
	t.Helper()
	_, err := Load(path)
	return err
}

// withHeader writes a profile whose tensors are sound and whose header is
// exactly the one given, so a refusal can only be about the header.
func withHeader(t *testing.T, header string) string {
	t.Helper()
	p := namedProfile("mine")
	path := filepath.Join(t.TempDir(), "v.safetensors")
	if err := p.Save(path); err != nil {
		t.Fatal(err)
	}
	entries := []safetensors.Entry{
		{Name: "speaker_embedding", Dtype: "F32", Shape: []int64{speakerDim},
			Data: safetensors.F32Bytes(p.SpeakerEmbedding)},
		{Name: "flow_embedding", Dtype: "F32", Shape: []int64{flowDim},
			Data: safetensors.F32Bytes(p.FlowEmbedding)},
		{Name: "prompt_tokens", Dtype: "I64", Shape: []int64{int64(len(p.PromptTokens))},
			Data: safetensors.I64Bytes(p.PromptTokens)},
		{Name: "prompt_mel", Dtype: "F32", Shape: []int64{melBins, 1},
			Data: safetensors.F32Bytes(p.PromptMel)},
		{Name: "cond_prompt_tokens", Dtype: "I64", Shape: []int64{int64(len(p.CondPromptTokens))},
			Data: safetensors.I64Bytes(p.CondPromptTokens)},
	}
	if err := safetensors.Write(path, entries, map[string]string{"voice": header}); err != nil {
		t.Fatal(err)
	}
	return path
}

// A header value of the wrong JSON type is refused by name, not coerced.
//
// `language: 5` read as the language id "5" here and as English in three of
// the other four ports, and language selects the text funnel: a voice whose
// header says 5 was spoken by whichever funnel the coercion happened to land
// on. VoiceProfile._header_str and _header_int refuse the five fields in these
// sentences.
func TestLoadRefusesAHeaderValueOfTheWrongType(t *testing.T) {
	for _, tc := range []struct{ header, want string }{
		{`{"format_version":1,"language":5}`, "voice header 'language' must be a string, got 5"},
		{`{"format_version":1,"language":null}`, "voice header 'language' must be a string, got None"},
		{`{"format_version":1,"language":true}`, "voice header 'language' must be a string, got True"},
		{`{"format_version":1,"language":["en"]}`, "voice header 'language' must be a string, got ['en']"},
		{`{"format_version":1,"name":5}`, "voice header 'name' must be a string, got 5"},
		{`{"format_version":1,"enrolment":5}`, "voice header 'enrolment' must be a string, got 5"},
		{`{"format_version":true}`, "voice header 'format_version' must be a number, got True"},
		{`{"format_version":"1"}`, "voice header 'format_version' must be a number, got '1'"},
		{`{"format_version":1.9}`, "voice header 'format_version' must be a whole number, got 1.9"},
		{`{"format_version":1,"source_sample_rate":"24000"}`,
			"voice header 'source_sample_rate' must be a number, got '24000'"},
		{`{"format_version":1,"source_sample_rate":24000.5}`,
			"voice header 'source_sample_rate' must be a whole number, got 24000.5"},
	} {
		err := errorFromLoad(t, withHeader(t, tc.header))
		if err == nil {
			t.Errorf("header %s was accepted", tc.header)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("header %s refused as %q, want %q", tc.header, err, tc.want)
		}
	}
}

// A header that is not a JSON object is refused by name rather than read as
// version zero, which blames a version the file does not carry.
func TestLoadRefusesAHeaderThatIsNotAnObject(t *testing.T) {
	for header, phrase := range map[string]string{
		"[1, 2]":  "an array",
		`"hello"`: "a string",
		"42":      "a number",
		"null":    "null",
	} {
		err := errorFromLoad(t, withHeader(t, header))
		if err == nil {
			t.Errorf("header %s was accepted", header)
			continue
		}
		want := "voice header is " + phrase + ", expected a JSON object"
		if !strings.Contains(err.Error(), want) {
			t.Errorf("header %s refused as %q, want %q", header, err, want)
		}
	}
}

// An absent field takes its default, which is the branch every profile
// written before a field existed goes down.
func TestLoadDefaultsEveryAbsentHeaderField(t *testing.T) {
	p, err := Load(withHeader(t, `{"format_version":1}`))
	if err != nil {
		t.Fatal(err)
	}
	// The file's own stem, which is the name the reference gives an unnamed
	// profile.
	if p.Name != "v" {
		t.Errorf("name %q, want the file stem", p.Name)
	}
	if p.Language != "en" || p.SourceSampleRate != 24000 || p.Enrolment != firstWindowEnrolment {
		t.Errorf("defaults are %q, %d, %q", p.Language, p.SourceSampleRate, p.Enrolment)
	}
}
