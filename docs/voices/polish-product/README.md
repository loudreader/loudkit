# Polish voices in everyday apps

Compare Gosia and Darkman on five messages an app might read aloud. The input
column shows exactly what the app supplied. The next column gives the words a
Polish listener should hear.

| App text | Expected reading | Gosia | Darkman |
|---|---|---|---|
| Cena biletu to 22 zł. | Cena biletu to dwadzieścia dwa złote. | [Listen](../preview/audio/polish-product/gosia-ticket-price.opus) | [Listen](../preview/audio/polish-product/darkman-ticket-price.opus) |
| Do zapłaty: 12,05 zł. | Do zapłaty: dwanaście złotych i pięć groszy. | [Listen](../preview/audio/polish-product/gosia-invoice-balance.opus) | [Listen](../preview/audio/polish-product/darkman-invoice-balance.opus) |
| Wybierz 2 z 3 ofert. | Wybierz dwie z trzech ofert. | [Listen](../preview/audio/polish-product/gosia-offer-selection.opus) | [Listen](../preview/audio/polish-product/darkman-offer-selection.opus) |
| Masz 5 nowych wiadomości. | Masz pięć nowych wiadomości. | [Listen](../preview/audio/polish-product/gosia-notification-count.opus) | [Listen](../preview/audio/polish-product/darkman-notification-count.opus) |
| Faktura numer 15 jest gotowa. | Faktura numer piętnaście jest gotowa. | [Listen](../preview/audio/polish-product/gosia-invoice-number.opus) | [Listen](../preview/audio/polish-product/darkman-invoice-number.opus) |

## Listening review

**Native-speaker review is pending for both voices.** For each clip, write down
the amount, count or instruction you hear on the first play. Compare it with the
expected reading on a second play. Record the voice and phrase for every missed
word, incorrect ending or unnatural pause. In particular, check *złote* after
22, the five groszy after the decimal comma, and *dwie z trzech* in the choice
of offers. A listener's verdict applies to that voice and that phrase.

## Automated results from 20 November 2026

The Polish text processor produced the five expected expansions above. Both
voices rendered all five cases as mono 24 kHz WAV. The ten published Opus
clips decoded to the same durations as their sources, ranging from 2.20 to
3.64 seconds.

Whisper small transcribed each source WAV independently on CPU. We specified
Polish and gave it no prompt. It recovered both amounts for both voices and
the notification count and invoice number in all four recordings. It rendered
the start of Gosia's ticket message as “Senamilet Tutu 22 zł.” and Darkman's
selection instruction as “Wybierz dwie stszech ofert.” Start with those two
clips when reviewing intelligibility. Digit-form transcripts also leave the
spoken currency endings for a listener to verify. [Raw transcripts, decoding
settings and file hashes](whisper-small.json)

## Render comparison recordings

The commands below render the same prompts with the installed SDK. The archived clips use
`loudreader/loudr-1` revision `7516673ee74ab2228ffc4e99223d296d8904d7ff`,
PyTorch 2.15.0 on an Apple Silicon CPU, text processor `funnel-7` and seed 7.

```sh
loudkit download loudreader/loudr-1 --for torch \
  --revision 7516673ee74ab2228ffc4e99223d296d8904d7ff --local-dir out/pl-model
for voice in gosia darkman; do
  python research/eval_roundtrip.py out/pl-model "$voice" "out/pl/$voice" --language pl
done
```

Each directory contains the five application messages, three pronunciation
probes and a manifest linking each WAV to its input and voice. The
[probe corpus](../../../tests/data/probes/probes.json) also records the
expected wording. For character-error scoring of these cases, transcribe the
spoken number words consistently with that reference.
