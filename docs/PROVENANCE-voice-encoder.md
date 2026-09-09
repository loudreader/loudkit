# Provenance: `ve.safetensors`

The utterance voice encoder that drives voice cloning. This page records what it
is, where it came from, and the terms it is redistributed under.

## The artefact

| field | value |
|---|---|
| file | `ve.safetensors` |
| size | 5,695,784 bytes |
| sha256 | `f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c` |
| contents | 16 tensors, `float32`: a 3-layer LSTM (`lstm.weight_ih_l{0,1,2}`, `lstm.weight_hh_l{0,1,2}`, biases) and a linear projection |
| what it does | 40-mel partials of a ≤10 s clip → a 256-d utterance embedding |
| used by | `loudkit.enroll` only. Synthesis does not touch it; a checkpoint without it speaks every voice in the library and cannot make a new one. |

## Where it comes from

It is the `VoiceEncoder` of the **Chatterbox T3** architecture, published by
**Resemble AI** under the **MIT licence**:

- <https://github.com/resemble-ai/chatterbox>
- <https://huggingface.co/ResembleAI/chatterbox>

The weights were not retrained. They were re-exported: read out of the upstream
artefact and written to safetensors by this project's own export script.

The same export path produced the other enrollment components named in
`NOTICE`: the speech tokenizer, the CAM++ speaker encoder, and the T3
conditioning layers. `NOTICE` lists their upstream licences.

## The licence it ships under

**MIT**, the same terms as the upstream weights.

That is the conservative reading. MIT permits sublicensing, so a derivative may
in principle be distributed under Apache-2.0, and this project's *code* is
Apache-2.0 for the express patent grant. Relicensing somebody else's MIT weights
means asserting terms over an artefact whose substance is theirs, and it buys
only tidiness in a table. The weights stay MIT, the code stays Apache-2.0, and
`README.md` says so where a redistributor reads it before `NOTICE`.

The full MIT text and Resemble AI's copyright line are in
[`NOTICE`](../NOTICE), which ships inside the pip, npm, crates.io and Go
packages rather than only in this repository.

## Verified upstream match

On 2026-09-05, all 16 tensor names, shapes, dtypes and values were compared with
[`ve.safetensors` at Chatterbox revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`](https://huggingface.co/ResembleAI/chatterbox/blob/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/ve.safetensors).
Every tensor matched exactly in the final local 0.1.1 bundle. The complete file
also has the same SHA-256 recorded above. This establishes an immutable upstream
source for the bytes we distribute.

The original export did not record its upstream checkout revision. The match
above establishes tensor identity with a published snapshot; it does not
reconstruct the original checkout or require identical safetensors headers.
