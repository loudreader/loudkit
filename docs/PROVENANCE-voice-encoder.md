# Provenance: `ve.safetensors`

`ve.safetensors` holds the utterance voice encoder that PyTorch enrollment
uses.

## The artefact

| field | value |
|---|---|
| file | `ve.safetensors` |
| size | 5,695,784 bytes |
| sha256 | `f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c` |
| contents | 16 tensors, `float32`: a 3-layer LSTM (`lstm.weight_ih_l{0,1,2}`, `lstm.weight_hh_l{0,1,2}`, biases), a linear projection, and the two similarity parameters `similarity_weight` and `similarity_bias` |
| what it does | 40-mel partials of a ≤10 s clip → a 256-d utterance embedding |
| used by | PyTorch enrollment (`loudkit.enroll` on a torch device). ONNX and CoreML enrollment use the exported voice-encoder graph instead. Synthesis does not use this file: without it, a checkpoint speaks every existing voice profile, but PyTorch cannot enroll a new one. |

## Where it comes from

It is the `VoiceEncoder` of the **Chatterbox** model, published by
**Resemble AI** under the **MIT licence**:

- <https://github.com/resemble-ai/chatterbox>
- <https://huggingface.co/ResembleAI/chatterbox>

The weights are the upstream weights, not retrained. This project's export
script read them from the upstream file and wrote them to safetensors.

The same export path produced the other enrollment components named in
`NOTICE`: the speech tokenizer, the CAM++ speaker encoder, and the
conditioning layers of the token generator. `NOTICE` lists their upstream
licences.

## The licence it ships under

**MIT**, the same terms as the upstream weights. MIT permits sublicensing, but
this project keeps the weights under MIT. The loudkit code is Apache-2.0.

The full MIT text and Resemble AI's copyright line are in
[`NOTICE`](../NOTICE). `NOTICE` ships inside the pip, npm, crates.io and Go
packages.

## Verified upstream match

On 2026-09-05, all 16 tensor names, shapes, dtypes and values in the 0.1.1
`ve.safetensors` were compared with
[`ve.safetensors` at Chatterbox revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`](https://huggingface.co/ResembleAI/chatterbox/blob/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/ve.safetensors).
Every tensor matched exactly. The upstream file also has the SHA-256 in the
table above.

The original export did not record its upstream checkout revision. The match
shows that the file is identical to a published upstream snapshot. It does not
identify the checkout that the export used.
