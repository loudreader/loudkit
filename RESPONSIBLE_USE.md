# Responsible use

loudkit clones a voice from a few seconds of audio and reads any text in it. It
runs locally, with no account and no network after the model download. loudkit
cannot verify who owns a voice sample. You are responsible for having the right
to use it.

## Allowed

- Your own voice, or a voice you have written permission to use.
- Public-domain and openly licensed recordings, within their terms.
- Accessibility: screen reading, assistive speech, restoring a voice someone has
  lost.
- Narration, localisation, dubbing, games, and research, with the consent of
  whoever the voice belongs to.
- The shipped voices are enrollments of recordings made or released for
  speech-technology use: personal donations recorded expressly for TTS (CC0),
  and CC0 / CC-BY corpora whose terms allow synthesis.
  [docs/voices/roster/provenance.json](docs/voices/roster/provenance.json)
  names the donor or source, licence and consent basis per voice. No audio
  without a stated licence ships here.

## Prohibited

- Impersonating a real person to deceive, defraud, or harass.
- Putting words in a public figure's mouth, including satire that is not clearly
  labelled as synthetic.
- Defeating voice authentication, or helping anyone do so.
- Cloning a voice from a recording published for another purpose (a podcast, a
  lecture, a video) without the speaker's consent. A public recording does not
  give consent.
- Distributing a voice profile of an identifiable person who has not consented.
  Consent to publish one clip does not cover a profile that reproduces the
  voice on demand for everyone who downloads it.

## Disclosure

Label synthetic audio as synthetic, and cite the source recording when you
publish a sample.

The EU AI Act's transparency obligations for synthetic content are in
[Article 50](https://eur-lex.europa.eu/eli/reg/2024/1689/oj?locale=en). It
requires providers of systems that generate synthetic audio to mark the output
in a machine-readable format, so that it is detectable as artificially
generated. The marking must be effective, interoperable, robust and reliable as
far as this is technically feasible. The Article does **not** name a format,
and it does not name C2PA. The Commission's
[transparency guidelines](https://digital-strategy.ec.europa.eu/en/policies/guidelines-transparency-ai-generated-content)
cover which obligations fall on whom, plus the exemptions and transitional
arrangements. C2PA Content Credentials are one widely used format for
machine-readable marking. loudkit does not use C2PA, and its own manifest
(below) is not a compliance verdict.

**None of this is legal advice. Shipping loudkit's provenance manifest does not
show that you comply.** Your obligations depend on whether you are a provider or
a deployer, and on how you use loudkit. Get qualified legal advice.

loudkit writes a marking by default. Every WAV that `Result.save` writes, and
every WAV the server returns, carries an **unsigned loudkit provenance
manifest** in JUMBF-shaped boxes. [What a saved WAV records](docs/reference/provenance.md)
gives where each writer puts it. One-shot HTTP replies also carry it in the
`X-Loudkit-Provenance` header. It holds the algorithm
fingerprint, the seed, and a SHA-256 that binds it to the audio bytes. WAVs from
the Go, Rust, JS and Swift ports carry no manifest.

**It is not C2PA.** A C2PA manifest is a signed manifest store. This is one
JSON document in boxes that borrow JUMBF's shape, and only loudkit's own tools
(`loudkit verify`) read it. See
[`docs/reference/provenance.md`](docs/reference/provenance.md). It is unsigned.
It records what made the file, and it does not identify a publisher. Deployers
can add their own signing and disclosure.

## What loudkit ships

We ship 28 voice profiles enrolled from recordings donated for speech
technology or released under terms that permit this use. Every profile has a
named source, licence and consent basis in the public roster. We do not ship
profiles made from private recordings or recordings published for an unrelated
purpose without the speaker's permission.

The enrollment code is included, for making a profile from a voice you have the
right to use.

The maintainers close issues and pull requests that ask for help with
undisclosed impersonation, voice authentication bypass, or stripping provenance
from generated audio.

These consent and attribution requirements apply to every backend and SDK,
including cloning on ONNX Runtime or CoreML without PyTorch, and to voice
profiles made in Python, Swift, Go, Rust or TypeScript. A voice profile works
with both models.
