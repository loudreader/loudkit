# Responsible use

loudkit clones a voice from a few seconds of audio and reads arbitrary text in
it, locally, with no account and no network. loudkit cannot verify who owns a
voice sample. That is your responsibility, and it does not transfer to us.

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
  names the donor or source, licence and consent basis per voice. No anonymous
  scraped audio ships here, and no voice this project cloned from a private
  individual ships either.

## Prohibited

- Impersonating a real person to deceive, defraud, or harass.
- Putting words in a public figure's mouth, including satire that is not clearly
  labelled as synthetic.
- Defeating voice authentication, or helping anyone do so.
- Cloning a voice from a recording published for another purpose (a podcast, a
  lecture, a video) without the speaker's consent. Public is not consenting.
- Distributing a voice pack of an identifiable person who has not consented.
  Publishing one clip is not the same act as publishing a file that reproduces
  someone's voice on demand, for everyone who downloads it.

## Disclosure

Label synthetic audio as synthetic, and cite the source recording when you
publish a sample.

On the law. The EU AI Act's transparency obligations for synthetic content sit
in [Article 50](https://eur-lex.europa.eu/eli/reg/2024/1689/oj?locale=en). It
requires providers of systems generating synthetic audio to mark the output in a
machine-readable way. It does **not** name a format, and it does not name C2PA.
The Commission's
[transparency guidelines](https://digital-strategy.ec.europa.eu/en/policies/guidelines-transparency-ai-generated-content)
cover which obligations fall on whom, plus the exemptions and transitional
arrangements. C2PA Content Credentials are a widely used way to satisfy a
machine-readable-marking requirement. They are a choice this project made, not
a compliance verdict it can hand you.

**None of this is legal advice, and shipping loudkit's manifest is not a
finding that you comply.** Whether you are a provider or a deployer, and what
that obliges, depends on facts about you. Ask someone qualified.

loudkit writes that marking by default. Every saved WAV (`Result.save`) and
every server response carries a **claim-only C2PA manifest** (a JUMBF `c2pa`
box, plus the `X-Loudkit-Provenance` header over HTTP) with the algorithm
fingerprint, the seed, and the SHA-256 binding it to the audio bytes. See
[`docs/reference/provenance.md`](docs/reference/provenance.md). It is unsigned,
because signing needs a certificate and that is the deployer's choice. It says
what made the file, not who vouches for it.

This is the machine-readable marking loudkit provides by default. Deployers can
add their own signing policy and disclosure around it.

## What we ship, and what we will not

We ship twenty voice profiles enrolled from recordings donated for speech
technology or released under terms that permit this use. Every profile has a
named source, licence and consent basis in the public roster. We do not ship
profiles made from private recordings or recordings published for an unrelated
purpose without the speaker's permission.

The enrollment code is included so you can make your own profile from a voice
you have the right to use.

Issues and pull requests asking for help with undisclosed impersonation, voice
authentication bypass, or stripping provenance from generated audio will be
closed.
