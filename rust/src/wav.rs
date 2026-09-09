//! WAV in and out, on `std` alone.
//!
//! Out: 16-bit PCM, mono, by the same quantisation rule every port uses:
//! `floor(x * 32768)` clipped to the int16 range (`loudkit.synthesis._quantise`).
//! Two ports that both "write a WAV" otherwise differ by one LSB on half the
//! samples, because some encoders floor and some round.
//!
//! In: 16-bit PCM and 32-bit float, so a reference clip recorded by anything
//! can be handed to [`crate::enroll`]. Multi-channel is averaged to mono, which
//! is what the Python path does through librosa.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// A mono waveform and the rate it is sampled at.
///
/// What a WAV holds, in the one shape that can be written or read.
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    /// Wrap rendered samples so they can be written.
    #[must_use]
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Audio {
            samples,
            sample_rate,
        }
    }

    /// Duration in seconds.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// Write a 16-bit PCM WAV to `path`.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be created or written, and when
    /// the waveform is too long for the 32-bit sizes a RIFF header carries.
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut out = BufWriter::new(file);
        self.write_wav(&mut out)?;
        // BufWriter drops its buffer on the floor if the flush fails, so the
        // flush is explicit: otherwise a full disk produces a truncated WAV and
        // an Ok.
        out.flush().map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Write a 16-bit PCM WAV to any sink.
    ///
    /// # Errors
    ///
    /// Returns an error when the sink refuses a write, and when the waveform is
    /// too long for the 32-bit sizes a RIFF header carries.
    pub fn write_wav<W: Write>(&self, w: &mut W) -> Result<(), String> {
        let data_len = u32::try_from(self.samples.len() * 2)
            .map_err(|_| "waveform is too long for a WAV (over 4 GB of samples)".to_string())?;
        let rate = self.sample_rate;
        let byte_rate = rate
            .checked_mul(2)
            .ok_or_else(|| format!("sample rate {rate} is not a rate"))?;
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&(36 + data_len).to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        header.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
        header.extend_from_slice(&1u16.to_le_bytes()); // PCM
        header.extend_from_slice(&1u16.to_le_bytes()); // mono
        header.extend_from_slice(&rate.to_le_bytes());
        header.extend_from_slice(&byte_rate.to_le_bytes());
        header.extend_from_slice(&2u16.to_le_bytes()); // block align
        header.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        header.extend_from_slice(b"data");
        header.extend_from_slice(&data_len.to_le_bytes());
        w.write_all(&header).map_err(|e| e.to_string())?;

        let mut frames = Vec::with_capacity(self.samples.len() * 2);
        for &x in &self.samples {
            frames.extend_from_slice(&quantise(x).to_le_bytes());
        }
        w.write_all(&frames).map_err(|e| e.to_string())
    }

    /// Read a mono waveform from a WAV file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, is not a WAV, or holds
    /// samples in a format other than 16-bit PCM or 32-bit float.
    pub fn read_wav(path: impl AsRef<Path>) -> Result<Audio, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// One float sample to one int16 frame.
///
/// `floor(x * 32768)`, clipped, not `round(x * 32767)`: it is what libsndfile's
/// WAV writer does and so what every loudkit WAV has always held. NaN casts to
/// 0 rather than to a rail.
fn quantise(x: f32) -> i16 {
    let scaled = (f64::from(x) * 32768.0).floor();
    if scaled.is_nan() {
        return 0;
    }
    scaled.clamp(-32768.0, 32767.0) as i16
}

/// Parse a RIFF/WAVE file into mono float samples.
fn decode(bytes: &[u8]) -> Result<Audio, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_string());
    }
    let mut format: Option<(u16, u16, u32)> = None; // (tag, bits, rate)
    let mut channels = 0u16;
    let mut data: Option<&[u8]> = None;
    let mut at = 12;
    // Chunks are walked rather than assumed at offset 36: recorders put LIST,
    // fact and JUNK chunks between `fmt ` and `data`, and a fixed offset reads
    // those as audio.
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
        let body_at = at + 8;
        let end = body_at.checked_add(size).ok_or("chunk size overflows")?;
        if end > bytes.len() {
            return Err(format!(
                "truncated: a chunk claims {size} bytes and the file has {}",
                bytes.len() - body_at
            ));
        }
        let body = &bytes[body_at..end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err("fmt chunk is too short".to_string());
            }
            let mut tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            // WAVE_FORMAT_EXTENSIBLE says nothing itself; the real tag is the
            // first two bytes of the sub-format GUID in the chunk's extension.
            if tag == 0xFFFE && body.len() >= 26 {
                tag = u16::from_le_bytes(body[24..26].try_into().unwrap());
            }
            format = Some((tag, bits, rate));
        } else if id == b"data" {
            data = Some(body);
        }
        // Chunks are word-aligned: an odd size is followed by a pad byte that
        // is not counted in it.
        at = end + (size & 1);
    }
    let (tag, bits, rate) = format.ok_or("no fmt chunk")?;
    let data = data.ok_or("no data chunk")?;
    if channels == 0 {
        return Err("fmt chunk claims zero channels".to_string());
    }
    let interleaved: Vec<f32> = match (tag, bits) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => {
            return Err(format!(
                "unsupported sample format (tag {tag}, {bits}-bit): this reader \
                 takes 16-bit PCM and 32-bit float. Convert with \
                 `ffmpeg -i in.wav -acodec pcm_s16le out.wav`"
            ))
        }
    };
    let n = usize::from(channels);
    let samples = if n == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(n)
            .map(|frame| frame.iter().sum::<f32>() / n as f32)
            .collect()
    };
    Ok(Audio {
        samples,
        sample_rate: rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_16_bit() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin() * 0.5).collect();
        let mut buf: Vec<u8> = Vec::new();
        Audio::new(samples.clone(), 24_000)
            .write_wav(&mut buf)
            .unwrap();
        assert_eq!(buf.len(), 44 + samples.len() * 2);
        let back = decode(&buf).unwrap();
        assert_eq!(back.sample_rate, 24_000);
        assert_eq!(back.samples.len(), samples.len());
        for (a, b) in samples.iter().zip(&back.samples) {
            // One quantisation step, and no more: 16-bit is the only loss.
            assert!((a - b).abs() <= 1.0 / 32768.0, "{a} vs {b}");
        }
    }

    /// A probe is a number, or the string "nan", which JSON cannot spell.
    fn probe(v: &serde_json::Value) -> f32 {
        match v.as_str() {
            Some("nan") => f32::NAN,
            Some(other) => panic!("unknown probe {other:?}"),
            None => v.as_f64().unwrap() as f32,
        }
    }

    /// Every value is what `loudkit.synthesis._quantise` answers for the same
    /// input, written by Python into the shared fixture.
    #[test]
    fn quantise_matches_the_shared_rule() {
        let Some(cases) = crate::shared_fixture("wav_quantise.json") else {
            return;
        };
        let cases = cases["cases"].as_array().unwrap();
        assert!(cases.len() >= 10, "the fixture holds probes");
        for c in cases {
            let x = probe(&c["x"]);
            assert_eq!(i64::from(quantise(x)), c["pcm16"].as_i64().unwrap(), "{x}");
        }
        assert_eq!(quantise(0.0020349235), 66); // the sample the rule was picked on
    }

    /// The header and the samples are the bytes Python writes.
    #[test]
    fn the_wav_bytes_are_the_fixtures() {
        let Some(cases) = crate::shared_fixture("wav_header.json") else {
            return;
        };
        let cases = cases["cases"].as_array().unwrap();
        assert!(cases.len() >= 3, "the fixture holds cases");
        for c in cases {
            let samples: Vec<f32> = c["samples"].as_array().unwrap().iter().map(probe).collect();
            let rate = c["sample_rate"].as_u64().unwrap() as u32;
            let mut buf: Vec<u8> = Vec::new();
            Audio::new(samples.clone(), rate)
                .write_wav(&mut buf)
                .unwrap();
            let hex = crate::hex(&buf);
            assert_eq!(
                hex,
                c["hex"].as_str().unwrap(),
                "{} samples at {rate} Hz",
                samples.len()
            );
        }
    }

    #[test]
    fn reads_float_wav_and_downmixes() {
        // Two channels of 32-bit float; the mono result is their average.
        let left = [0.25f32, -0.5, 0.75];
        let right = [0.75f32, 0.5, 0.25];
        let mut body: Vec<u8> = Vec::new();
        for i in 0..3 {
            body.extend_from_slice(&left[i].to_le_bytes());
            body.extend_from_slice(&right[i].to_le_bytes());
        }
        let wav = build_wav(3, 32, 2, 16_000, &body);
        let got = decode(&wav).unwrap();
        assert_eq!(got.sample_rate, 16_000);
        assert_eq!(got.samples, vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn skips_chunks_between_fmt_and_data() {
        let body = 1234i16.to_le_bytes().to_vec();
        let mut wav = build_wav(1, 16, 1, 8_000, &body);
        // Splice a 3-byte LIST chunk (odd, so it carries a pad byte) in front
        // of `data`, which is where every recorder puts its metadata.
        let data_at = wav.windows(4).position(|w| w == b"data").unwrap();
        let mut chunk = b"LIST".to_vec();
        chunk.extend_from_slice(&3u32.to_le_bytes());
        chunk.extend_from_slice(b"abc\0");
        wav.splice(data_at..data_at, chunk);
        let got = decode(&wav).unwrap();
        assert_eq!(got.samples.len(), 1);
        assert!((got.samples[0] - 1234.0 / 32768.0).abs() < 1e-9);
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        assert!(decode(b"not a wav at all").is_err());
        // 24-bit PCM is a real format and not one of the two.
        let wav = build_wav(1, 24, 1, 16_000, &[0, 0, 0]);
        let Err(err) = decode(&wav) else {
            panic!("24-bit PCM was accepted");
        };
        assert!(err.contains("24-bit"), "{err}");
    }

    fn build_wav(tag: u16, bits: u16, channels: u16, rate: u32, body: &[u8]) -> Vec<u8> {
        let block = u32::from(channels) * u32::from(bits) / 8;
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(36 + body.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * block).to_le_bytes());
        out.extend_from_slice(&(block as u16).to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        out
    }
}
