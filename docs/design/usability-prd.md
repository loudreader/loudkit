# Product requirements for offline speech and application integration

## User outcomes

A reader can choose a voice in their language, hear a passage locally and save
a playable recording. An application developer can ship the same selected
model, start reading a long chapter promptly and tell whether the speech service
can handle a request. Someone evaluating loudkit on GitHub can try their own
sentence and see which products actually use the engine.

For release acceptance, prioritize failures that can lose an existing recording,
silently load the wrong weights or leave a screen reader silent. Next verify
offline setup, readiness and host application compatibility. English is the
first listening target for long-form narration. Check shared text changes in
every shipped voice language, and use language-specific samples for rules
that depend on local spelling or grammar.

## Requirements and acceptance

| User task | Required behavior | Acceptance check |
| --- | --- | --- |
| Use a previously downloaded model | Python and the local server can open voices in both Hugging Face cache layouts. An explicitly requested revision must match the cached release in every SDK when offline. | Open a named voice from a shared blob store. With the network unavailable and revision A installed, requesting B fails and requesting A succeeds. |
| Pick a voice in a language | The CLI can list installed voices and their languages offline, filter by language, and provide names suitable for scripts. | A cached official release yields 28 names. Filtering for Polish finds `gosia` and `darkman`. Verify local directories and custom profiles as well. |
| Preview and export audio | Playback alone uses a temporary file. An explicit output path is kept. Python's WAV writer and CLI validate their supported output format. | Start with an existing `out.wav`, preview speech and verify its contents survive. Decode the samples of an explicitly saved WAV. An unsupported extension fails before writing, including when metadata is disabled. |
| Read a chapter | All five SDKs preserve word boundaries across line wraps, use plausible breaks in long sentences and keep short sentence endings together when they fit the model. Splitting time scales with the length of a chapter. | Compare the prepared text, chunk boundaries and actual audio of long English sentences before and after a change. Exercise dialogue, paragraph breaks, slow voices and custom budgets. Run shared conformance across SDKs. |
| Narrate everyday text | Titles, times, prices and grouped numbers keep their meaning when read aloud. | Start with English examples in the public text API. Check space-grouped quantities in each shipped voice language, including numbers copied from PDFs with narrow nonbreaking spaces. |
| Read Polish notifications, quantities and prices | The prepared text handles the reviewed constructions in context. Both Polish voices have repeatable listening examples for the use cases. | Verify “Wybierz 2 z 3 ofert”, prices written with decimal commas and złoty/grosz agreement, weights and areas. Render the examples on both voices. Record automated results separately from native listening judgments. |
| Navigate using a screen reader | Stopping speech invalidates the previous utterance. The newest focus can still be spoken when older renders are finishing. Standard Speech Dispatcher settings select a profile using the requested name or language and voice. | Send three rapid focus changes with intervening stops. Hear only the current selection once the earlier work is obsolete. Each request receives one accurate status. Select English `joe` and Polish `darkman` via the module protocol. |
| Start the local server | A deployment signals readiness only when its engine and at least one usable voice are available. It can recover when the operator fixes the voice directory. | Exercise the actual HTTP and container probe path with empty, invalid and usable profiles. An operator can identify the reason for a failed probe. |
| Clone a voice in an app | JS validates the source recording before fetching expensive enrollment assets. Apple apps can provide source or compiled Core ML enrollment models, including mixed directories. | A missing or invalid recording triggers no JS model download. A Swift bundle can resolve all three enrollment stages with either supported extension. |
| Embed loudkit in an existing process | Python documents its PyTorch process-wide settings at the point of integration and provides a way for the host to retain its settings. Node can import the packaged SDK on its declared minimum runtime. | Start with non-default PyTorch flags, load the engine in host-controlled mode and check that the flags survive. Import the packaged JS entry point on the minimum Node version. |
| Evaluate loudkit before installing | The homepage links to a working own-text experience from the voice player. SDK comparisons show measured time to first audio with hardware, model and workload. Batched generator throughput is labelled with its measurement scope. | Follow the demo link from the homepage. Trace each performance figure to the published benchmark and confirm that generator throughput cannot be mistaken for the speed of a complete render. |
| Find implementations on GitHub | The root README names verified projects, explains the role loudkit plays and links to evidence for each. Readers can distinguish a shipped application from a developer preview. | Follow each showcase link to the implementation or project. A provider configuration alone does not establish that its host application ships loudkit. |
| Maintain the integrations | Dependency declarations keep their constraints and concise rationale. Shared fixtures and tests exercise the user-facing operations above. | Compare parsed package metadata before and after the comment cleanup. Run the affected Python, native SDK and site checks. |

## Compatibility and evidence

The shared identity contract governs audible changes. Pin the text preparation
rules in fixtures across all five SDKs and record any fingerprint change.
Historical performance measurements and published voice samples retain the
identities and versions under which they were produced.
[Identity contract](../reference/IDENTITY-CONTRACT.md)

Use the actual cached model for Python synthesis and a real decoder for WAV
acceptance. Deterministic test doubles are appropriate for an unavailable Hub,
an interrupted render and the Speech Dispatcher protocol. Device playback,
native-speaker listening and hardware-specific inference each require their
own evidence. [Evaluation protocol](evaluation.md)

### Acceptance measurements

Splitting the same passage before and after the change produced the following
elapsed times in milliseconds. Every run preserved the original chunk boundaries.

| SDK | 50,000 characters | 100,000 characters | 200,000 characters |
| --- | ---: | ---: | ---: |
| Go | 117.86 → 1.17 | 510.16 → 2.74 | 1706.93 → 5.61 |
| Node | 157.33 → 2.04 | 637.44 → 3.17 | 3115.98 → 8.96 |
| Rust | 21.74 → 0.14 | 85.14 → 0.23 | 310.64 → 0.44 |
| Swift | 38.75 → 20.76 | 120.16 → 63.88 | 225.67 → 94.54 |

Go, Node and Swift used repetitions of “A sentence of plain text. ”, truncated
to the exact length. Rust used a mixed Polish and English passage. Go reports
the median of five runs, Swift the median of three warm runs, and Node and
Rust one timed run each. These samples show lower elapsed times at each size.
They do not establish an asymptotic bound for Swift or characterize run-to-run variance.
The measurements cover text splitting on a shared
development machine. For measured streaming latency, see the
[end-to-end benchmarks](../benchmarks.md#end-to-end-on-apple-m3-pro-011-both-models).

At the end of the initial usability review, all five SDKs passed 228 shared
text cases. The current shared count is reported in the [parity table](../parity-measured.md). The
[Polish listening panel](../voices/polish-product/README.md) pairs each
application message with its intended reading and a recording from each
voice, and tracks the status of native-speaker review.

The offline Python CLI generated a 24 kHz WAV using Gosia from the shared
Hub cache. With the official model loaded, the HTTP service returned 503
for an empty voice directory, then returned 200 and synthesized speech
after a valid profile was added. The Docker health command was run against
that service outside the container. Speech Dispatcher acceptance used the
module subprocess, a local HTTP server and a test player. A desktop Orca
session and the native Polish listening review remain open for release QA.

## Follow-up audit acceptance criteria

| Priority | User journey | Regression criterion |
| --- | --- | --- |
| P1 | Narrate a leading decimal | .75% and 0.75% have identical spoken forms across SDKs. Preserve signs, currency and sentence punctuation. |
| P1 | Download during a branch update | Resolve once. Fetch the listing and every file by that commit before recording the receipt. An offline exact pin returns the same bytes. |
| P2 | Continue a Turbo passage | Unary and streamed responses provide the same continuation as the engine after an odd-length window. |
| P2 | Read HTML and technical prose | Decode escaped apostrophes and ampersands. Preserve both comparisons in x<low or x>high. |
| P2 | Render a chapter in Go or Rust | Assemble the returned spectrogram in linear time with unchanged bin order. |
| P2 | Add an offline backend | Failed preflight preserves the receipt for the installed backend. |
| P2 | Load a local Go model | A local directory named loudr-1 takes precedence over the Hub alias. |
| P2 | Export or verify WAV | The RIFF length includes provenance. Standard metadata after audio verifies as an unmarked file. Legacy marked files remain readable. |
| P2 | Choose a server voice | An unreadable profile is omitted or identified as unavailable with a structured error. |
| P2 | Enroll a short clip | Reject invalid audio before resolving weights in every enrollment mode. |
| P2 | Interrupt Node speech | Cancellation during mel decoding skips vocoding. |
| P3 | Profile synthesis | Swift reports measured stage times. Python CLI reports playback speed using elapsed wall time when stages overlap. |

### Integration behavior

The narration frontend supplies the omitted zero in leading decimals before
expanding currency and units. It decodes common HTML entities after stripping
tags so an escaped angle bracket remains text. For Turbo, the continuation
belongs to the last acoustic window because each window restarts decoder
pairing. Chain requests with the returned continuation from either transport.

A download receipt records both the requested revision and the resolved
commit. After resolving a revision, fetch its release inventory and blobs by
commit. A preflight failure leaves a previously verified local release usable.
The Go loader accepts local model directories named after either model alias.

Exported WAVs carry the provenance manifest inside the declared RIFF size.
The reader also accepts trailers from earlier Loudkit versions and skips
ordinary RIFF metadata. During long-form synthesis, the Go and Rust SDKs
copy each spectrogram row into its final position. Stage durations describe
work in each stage. Python's reported real-time factor uses elapsed time
when pipeline stages overlap.

### Running the follow-up regressions

The five SDKs consume the same [narration fixture](../../tests/data/conformance/speechtext.json).
The [Python frontend tests](../../tests/test_speechtext.py) also compare the
spoken forms of leading decimals against their explicit-zero equivalents.
[Hub regressions](../../tests/test_hub.py) exercise a moving branch and a
failed backend preflight using local fixtures.

[HTTP tests](../../tests/test_server.py) cover Turbo continuation, the WAV
header and unreadable profiles. [WAV tests](../../tests/test_provenance.py)
cover standard metadata and the legacy trailer. The [Go chapter test](../../go/engine/mel_test.go)
checks allocation count independently of machine speed. The [Python timing
test](../../tests/test_synthesis.py) drives a multi-window render with a
controlled elapsed time. These cases run with the ordinary SDK test suites.

### Follow-up review checks

The shipping model reached its 255-token cap on the 148-character Polish
counting passage in the shared chunk fixture with Darkman at seeds 1 and 7.
The planner now uses an in-budget pause when at least two eligible internal
pauses compete with a sentence ending beyond the target. English readings
with a short introductory comma can still finish their sentence in one window.
The two planned Polish windows used 222 and 69 tokens at seed 1, and 182 and 74
at seed 7. The German and English control passages fit at both seeds. These probes
exercise overflow recovery and do not measure listener preference or latency.

The catalog and readiness routes use the same validated profile list during
normal service. Validation is cached for the current revision of each installed
profile; a changed profile is retried and removed profiles drop out of the cache.

The sign-off regressions cover angle-bracket placeholders in command examples,
encoded alternatives in forms, assigned framework attributes, and adjacent
comparisons in English, Polish and German. A Polish question ending in
“dobrze?” exercises the planner when the last eligible comma would leave
one word. HTTP startup primes the voice catalog before serving readiness
requests. Historical graph-export tests cover both model families through
`funnel-9` in all five SDKs.

Unquoted HTML values end at whitespace. Regression cases include a URL query
with an equals sign, adjacent quoted attributes, and variable assignments
between two comparisons in prose. The inequality cases retain every word.
