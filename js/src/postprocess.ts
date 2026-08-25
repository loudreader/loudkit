/**
 * Deciding where a generated chunk actually ended.
 *
 * Mirrors `loudkit.postprocess`. This is a **detector**, not a filter: it reads
 * the speech tokens a chunk produced, answers one question — where did the
 * sentence really stop? — and returns a verdict. It never touches a sample of
 * audio.
 *
 * The artifact it removes is generated, not spectral. The decoder is
 * free-running, and silence tokens are exempt from both the repetition penalty
 * and the `min_p` cutoff (penalising silence measurably removes pauses), so once
 * the sentence is over those tokens keep probability mass indefinitely. The
 * decoder free-runs silence, and any step where a non-silence token survives the
 * cutoff becomes a hallucinated word — heard as "it finished, then a long gap,
 * then one random word".
 *
 * Every constant came from a device trace or a regression, and every rule is
 * pinned by `tests/data/conformance/postprocess.json`, which all five ports run.
 * Provenance is in `docs/reference/postprocess.md`.
 * Python reference: `loudkit/frontend/postprocess.py`.
 */

/**
 * What the engine does with a verdict.
 *
 * `trim` applies the cut, which changes the audio and therefore travels in the
 * fingerprint like every other audible decision. `report` runs the detectors and
 * attaches the verdict without acting on it. `off` skips them entirely.
 */
export type PostprocessMode = "off" | "report" | "trim";

export const POSTPROCESS_MODES: PostprocessMode[] = ["off", "report", "trim"];

/** Which rule fired. `clean` means none did. */
export type Reason =
  | "clean"
  | "dropout"
  | "repetition"
  | "silence_tail"
  | "terminal_echo"
  | "desperation"
  | "ended_tail";

/**
 * The detector constants. Algorithm layer: a port that uses a different
 * number produces different audio, so these are hashed into the fingerprint
 * rather than left as module constants.
 */
export interface PostprocessConfig {
  mode: PostprocessMode;

  /**
   * Hard stop for generation, as a multiple of the text-token count.
   *
   * Device trace of the showcase render: `t3.overrun gen=92 ceiling=92
   * bestEOS=74@0.003 floor=31` — ~26 text tokens stopped only because it hit the
   * ceiling, mid-sentence, already at 3.5 speech tokens per text token. NOT the
   * chunker's 2.6: there, guessing high only wastes window; here, guessing low
   * cuts a sentence off.
   */
  ceilingSpeechPerTextToken: number;
  /** Carries the very short texts, where a ratio alone is unsafe (1.6 s). */
  ceilingSlackTokens: number;

  /** Share of a tail that must be silence before it counts as one. */
  trailingFillerThreshold: number;
  /**
   * An unbroken silence run marking a structural boundary (~0.5 s at 25 Hz).
   *
   * A hallucinated word sits *behind* such a seam; under the share test alone
   * its burst lowers the silence ratio below threshold, so the ugliest tails
   * are exactly the ones the rescue refuses to cut.
   */
  trailingSilenceRunTokens: number;
  /**
   * Top of the stop-peak acceptance band in {@link desperationCut}, as a
   * multiple of the text-token count.
   *
   * Measured reads run 1.75–2.35 speech tokens per text token, so the band
   * reaches past every legitimate ending while staying well under the 4.5x
   * garbage threshold.
   */
  desperationBandRatio: number;
  /**
   * Slack above the proportional band, in speech tokens (~0.5 s). Carries the
   * short texts, where the ratio alone would close the band on endings a
   * legitimate read had already reached.
   */
  desperationBandFloor: number;
  /**
   * How confident the best stop must be before the share/run test is consulted
   * at all. EOS-defence bench, variant B.
   */
  fillerMinEosProbability: number;
  /**
   * How much speech may follow a seam and still be a hallucinated word rather
   * than a continuing clause (~0.4 s).
   *
   * Deliberately separate from `endedTailWordMax` despite holding the same
   * number: they govern different rows, so loosening
   * the trim on terminal chunks must not silently loosen this.
   */
  fillerMaxSpeechAfterRun: number;

  /**
   * Past this ratio the row certainly contains garbage, whatever its stop
   * confidence said.
   *
   * "It was as he expected." — 14 text tokens — came back as 96 speech tokens of
   * sentence-then-dense-babble, with the stop peak at the right *place* (45) but
   * confidence 0.000, so every probability-gated rescue refused. Real speech runs
   * 1.75–2.35 speech tokens per text token.
   */
  desperationSpeechPerTextToken: number;
  /**
   * Tiny texts are exempt: fixed overheads (breath, final pause) give a clean
   * "No!" a ratio of 6+ by itself.
   */
  desperationMinTextTokens: number;

  /** Silence before a blip that counts as stranding it (~0.24 s). */
  endedTailSilenceRun: number;
  /** <= 80 ms of "speech" is a click, not a word. */
  endedTailBlipMax: number;
  /**
   * A stray word behind a full seam on a *terminal* chunk is cut with it.
   * Continuation chunks keep their tails — their pauses are the sentence's
   * rhythm and their "end" is not an end.
   */
  endedTailWordMax: number;
  /** Pause left in place after trimming (~0.2 s). */
  endedTailKeep: number;

  /**
   * The ordinary terminal echo: a confident stop, late, with at most ~1.2 s
   * after it. The position rule keeps a real clause pause from reading as an
   * ending.
   */
  echoStrongEosProbability: number;
  echoStrongMaxTail: number;
  echoStrongMinPositionPct: number;

  /**
   * The narrow second path, for one regression ("...but a brigand. Pass.
   * Four.": `gen=124/124, bestEOS=109@0.004`). Confidence this weak is accepted
   * only with every corroborator at once.
   */
  echoWeakEosProbability: number;
  echoWeakMaxTail: number;
  echoWeakMinPositionPct: number;

  /** How many re-rolls a condemned window may get before shipping as is.
   * Only dropout and suspect retry; each attempt draws a derived seed. */
  retryMaxAttempts: number;

  /** How far a chunk's pace may drift from the passage's median before it is
   * flagged (multiplicative, both directions). */
  pacingTolerance: number;

  /**
   * Longest cycle, in tokens (~0.5 s), that counts as a stuck decoder. Above it
   * a repeated block is a phrase, and a repeated phrase is rhetoric.
   */
  repetitionMaxPeriod: number;

  /**
   * How many consecutive identical cycles a loop needs. Two is a repeated
   * phrase; three is necessary, not sufficient.
   */
  repetitionMinCycles: number;

  /**
   * How many tokens the repeating region must cover (~1.0 s). The constant that
   * does the work: measured across 27 renders in nine languages, a healthy row
   * repeats for at most 10 tokens. Cycle count alone fired on 22 of those 27.
   */
  repetitionMinSpan: number;

  /**
   * Early truncation: the row is too short to be the text it was asked for.
   * Reported, never cut — there is nothing to cut, and it is the most damaging
   * failure in the set because a listener cannot hear that content is absent.
   * The 25-token floor is the published criterion for a catastrophic
   * neural-codec TTS failure; the proportional test exempts a genuinely short
   * line, since the shortest healthy reads measured run 35 tokens.
   */
  dropoutMinTokens: number;
}

/** The shipping detector configuration. */
export const PRODUCTION_POSTPROCESS: PostprocessConfig = {
  mode: "trim",
  ceilingSpeechPerTextToken: 4.0,
  ceilingSlackTokens: 40,
  trailingFillerThreshold: 0.7,
  trailingSilenceRunTokens: 12,
  desperationBandRatio: 2.6,
  desperationBandFloor: 12,
  fillerMinEosProbability: 0.05,
  fillerMaxSpeechAfterRun: 10,
  desperationSpeechPerTextToken: 4.5,
  desperationMinTextTokens: 10,
  endedTailSilenceRun: 6,
  endedTailBlipMax: 2,
  endedTailWordMax: 10,
  endedTailKeep: 5,
  echoStrongEosProbability: 0.1,
  echoStrongMaxTail: 30,
  echoStrongMinPositionPct: 68,
  echoWeakEosProbability: 0.003,
  echoWeakMaxTail: 16,
  echoWeakMinPositionPct: 85,
  retryMaxAttempts: 2,
  pacingTolerance: 1.6,
  repetitionMaxPeriod: 12,
  repetitionMinCycles: 3,
  repetitionMinSpan: 24,
  dropoutMinTokens: 25,
};

/** What the detectors concluded about one chunk. */
export interface Inspection {
  /**
   * How many leading tokens survive — equal to the input length when nothing
   * fired, so a caller can always slice by it without branching.
   */
  keep: number;
  reason: Reason;
  /**
   * The row is impossibly long for its text and no anchor agreed where to cut.
   * Not an error and not a cut: a report. Shipping such a row silently is how
   * the artifact reached listeners in the first place.
   */
  suspect: boolean;
}

/** Everything the detectors need to know about one generated chunk. */
export interface InspectRequest {
  /** The denominator of every ratio rule. */
  textTokenCount: number;
  /** The EOS floor this row was generated under. */
  minTokens: number;
  /**
   * Step at which the stop token was most probable, or negative if it was never
   * observed.
   */
  eosPeakAt: number;
  eosPeakProb: number;
  /** Whether generation stopped at the stop token rather than a cap. */
  ended: boolean;
  /**
   * Whether this chunk ends the passage. A continuation chunk has no sentence
   * end, so its stop peak means nothing.
   */
  isTerminal: boolean;
  /** Whether generation was stopped by the length ceiling. */
  hitCeiling: boolean;
}

/**
 * Speech tokens at which the decoder is stopped whatever it thinks.
 *
 * Applied *during* generation: the tokens past it cost real time on a device and
 * are certain to be discarded. It only ever stops a row that was going to run
 * away — a model that stops on its own never reaches it.
 */
export function ceilingFor(
  textTokenCount: number,
  cfg: PostprocessConfig,
  window: number
): number {
  const proportional = Math.trunc(textTokenCount * cfg.ceilingSpeechPerTextToken);
  return Math.min(window, proportional + cfg.ceilingSlackTokens);
}

function silenceFlags(tokens: number[], silence: Iterable<number>): boolean[] {
  const set = new Set(silence);
  return tokens.map((t) => set.has(t));
}

/**
 * Whether what follows `index` is a trailing tail rather than more sentence.
 *
 * The overrun rescue cuts back to where the model came closest to stopping, and
 * that peak is a hint, not a verdict. Trusting it alone truncated whole
 * sentences: a voice reading a language its tag does not match may never commit
 * to stopping, so its best moment of hesitation lands a third of the way in.
 *
 * So the peak is corroborated by *what it proposes to discard* — either the tail
 * is mostly silence by share, or it holds a long unbroken run with only a stray
 * word behind it. Without that second half, a rhetorical pause mid-tail (25
 * silent tokens, then 80 of speech) matched the run rule and the rescue cut the
 * rest of the sentence off.
 */
/**
 * Indices of chunks whose pace drifts past the tolerance from the median.
 * Long-form drift: per-chunk pace (speech tokens / text tokens) against the passage's own median, report-only. The median rather than the mean, so one broken chunk cannot drag the baseline toward itself and hide.
 */
export function pacingOutliers(
  ratios: readonly number[],
  cfg: PostprocessConfig
): number[] {
  if (ratios.length < 3) {
    // One chunk has no neighbours; two cannot say which of them drifted.
    return [];
  }
  const ordered = [...ratios].sort((a, b) => a - b);
  const mid = Math.floor(ordered.length / 2);
  const median =
    ordered.length % 2 === 0 ? (ordered[mid - 1] + ordered[mid]) / 2 : ordered[mid];
  if (median <= 0) return [];
  const out: number[] = [];
  ratios.forEach((ratio, i) => {
    if (ratio > median * cfg.pacingTolerance || ratio < median / cfg.pacingTolerance) {
      out.push(i);
    }
  });
  return out;
}

/**
 * Whether the row is too short to be the text it was asked for.
 *
 * Two conditions, both required. The absolute floor catches a row that stopped
 * almost immediately whatever the text was; the proportional one keeps a
 * genuinely short line exempt, because a read producing less than one speech
 * token per text token has not said the text under any pronunciation.
 */
export function isDropout(
  tokenCount: number,
  textTokenCount: number,
  cfg: PostprocessConfig
): boolean {
  if (tokenCount >= cfg.dropoutMinTokens) return false;
  return textTokenCount > 0 && tokenCount < textTokenCount;
}

/**
 * Where a stuck decoder started looping, or null.
 *
 * The failure the tail rules cannot see, because it happens *inside* the row.
 * The mechanism is the one behind the trailing hallucinated word — the model's
 * own output becomes its context — but it strikes mid-sequence, so no rule that
 * reads the end can find it.
 *
 * Deliberately hard to trigger, because it is the only rule here that cuts
 * mid-sequence: a short cycle, repeated many times, matched exactly. A decoder
 * that has genuinely locked up emits the same tokens rather than similar ones,
 * and a fuzzy match on a signal this destructive would truncate real speech.
 *
 * A cycle that is entirely silence is never a loop — silence repeating is what
 * silence is, and the tail rules already judge pauses against where they sit.
 *
 * Returns one full cycle past the loop's start: the first instance is plausibly
 * the word the sentence wanted.
 */
export function repetitionCut(
  tokens: readonly number[],
  silence: Iterable<number>,
  cfg: PostprocessConfig
): number | null {
  const n = tokens.length;
  if (n < cfg.repetitionMinSpan) return null;
  const quietIds = new Set(silence);
  const quiet = tokens.map((t) => quietIds.has(t));

  // Earliest loop wins: a row that locks up twice locked up first at the first
  // one, and everything after it is already inside the failure.
  let best: number | null = null;
  const longestPeriod = Math.min(cfg.repetitionMaxPeriod, Math.floor(n / cfg.repetitionMinCycles));
  for (let period = 1; period <= longestPeriod; period++) {
    for (let start = 0; start + period * cfg.repetitionMinCycles <= n; start++) {
      let cycles = 1;
      for (let at = start + period; at + period <= n; at += period) {
        let same = true;
        for (let i = 0; i < period; i++) {
          if (tokens[at + i] !== tokens[start + i]) {
            same = false;
            break;
          }
        }
        if (!same) break;
        cycles++;
      }
      let allQuiet = true;
      for (let i = 0; i < period; i++) {
        if (!quiet[start + i]) {
          allQuiet = false;
          break;
        }
      }
      if (cycles >= cfg.repetitionMinCycles && cycles * period >= cfg.repetitionMinSpan && !allQuiet) {
        const candidate = start + period;
        if (best === null || candidate < best) best = candidate;
        break;
      }
    }
  }
  return best;
}

export function isTrailingFiller(
  tokens: number[],
  index: number,
  silence: Iterable<number>,
  cfg: PostprocessConfig
): boolean {
  if (index < 0 || index >= tokens.length) return false;
  const flags = silenceFlags(tokens.slice(index), silence);

  let silent = 0;
  let run = 0;
  let longestRun = 0;
  for (const isSilent of flags) {
    if (isSilent) {
      silent += 1;
      run += 1;
      if (run > longestRun) longestRun = run;
    } else {
      run = 0;
    }
  }
  if (silent / flags.length >= cfg.trailingFillerThreshold) return true;
  if (longestRun < cfg.trailingSilenceRunTokens) return false;

  // Collect qualifying runs, then require every gap of speech between them —
  // and after the last — to be a stray word or less. [seam][real
  // sentence][seam][word] fails: the tokens between the two seams are the
  // sentence itself, not filler trailing the first boundary.
  const runs: Array<[number, number]> = [];
  let scanRun = 0;
  let scanStart = 0;
  for (let i = 0; i < flags.length; i += 1) {
    if (flags[i]) {
      if (scanRun === 0) scanStart = i;
      scanRun += 1;
      if (scanRun === cfg.trailingSilenceRunTokens) runs.push([scanStart, i + 1]);
    } else {
      scanRun = 0;
    }
  }
  if (runs.length === 0) return false;
  if (runs[0][0] > cfg.fillerMaxSpeechAfterRun) return false;
  const last = runs[runs.length - 1];
  if (flags.length - last[1] > cfg.fillerMaxSpeechAfterRun) return false;
  for (let i = 1; i < runs.length; i += 1) {
    if (runs[i][0] - runs[i - 1][1] > cfg.fillerMaxSpeechAfterRun) return false;
  }
  return true;
}

/**
 * The rescue for rows whose *length* is the evidence. Returns the token count to
 * keep, or `null`.
 *
 * Past the ratio the row is certainly broken, so the question is where to cut,
 * not whether: at the first long silence run that starts past the floor (a run
 * straddling the floor belongs to the sentence, which is why the run's *start*
 * is tested), else at the stop peak if it sits in a band a real read could have
 * ended in. The band protects the mislabeled-language case (92 generated / 26
 * text = 3.5x), whose kind of row must never be cut at a peak landing a third of
 * the way in.
 *
 * `peakAllowed` is false for a continuation chunk: it has no sentence end, so
 * its stop peak means nothing.
 */
export function desperationCut(
  tokens: number[],
  textTokenCount: number,
  minTokens: number,
  eosPeakAt: number,
  silence: Iterable<number>,
  cfg: PostprocessConfig,
  peakAllowed = true
): number | null {
  if (textTokenCount < cfg.desperationMinTextTokens) return null;
  if (tokens.length < textTokenCount * cfg.desperationSpeechPerTextToken) return null;

  const earliest = Math.max(minTokens, 10);
  const flags = silenceFlags(tokens, silence);

  let runStart = -1;
  let run = 0;
  for (let i = 0; i < flags.length; i += 1) {
    if (flags[i]) {
      if (run === 0) runStart = i;
      run += 1;
      if (run >= cfg.trailingSilenceRunTokens && runStart >= earliest) return runStart;
    } else {
      run = 0;
    }
  }

  // No seam — the babble is dense; fall back to the model's own best stop, if
  // it lands where a real read could have ended.
  if (!peakAllowed) return null;
  const bandTop = Math.trunc(cfg.desperationBandRatio * textTokenCount) + cfg.desperationBandFloor;
  if (eosPeakAt >= earliest && eosPeakAt <= bandTop && eosPeakAt < tokens.length) {
    return eosPeakAt;
  }
  return null;
}

/**
 * Dead air past the sentence on a row that stopped when it meant to. Returns the
 * token count to keep, or `null`.
 *
 * Walked backward as `[sentence][r1 silence][burst][r2 silence]`. Three shapes
 * come off: a bare silence run half a second long; a silence run with a 1–2 token
 * blip right before the stop (the device specimen ended `.......#`); and, on a
 * *terminal* chunk only, a stray word behind a full seam.
 */
export function endedTailTrim(
  tokens: number[],
  silence: Iterable<number>,
  cfg: PostprocessConfig,
  isTerminal = false
): number | null {
  const flags = silenceFlags(tokens, silence);
  let j = tokens.length - 1;

  let r2 = 0;
  while (j >= 0 && flags[j]) {
    r2 += 1;
    j -= 1;
  }
  if (j < 0) return null;
  if (r2 >= cfg.trailingSilenceRunTokens) {
    const n = j + 1 + Math.min(r2, cfg.endedTailKeep);
    return n < tokens.length ? n : null;
  }

  let burst = 0;
  while (j >= 0 && !flags[j]) {
    burst += 1;
    j -= 1;
  }
  let r1 = 0;
  while (j >= 0 && flags[j]) {
    r1 += 1;
    j -= 1;
  }
  if (j < 0) return null; // the "burst" was the sentence

  const strandedClick = burst <= cfg.endedTailBlipMax && r1 >= cfg.endedTailSilenceRun;
  const strandedWord =
    isTerminal && burst <= cfg.endedTailWordMax && r1 >= cfg.trailingSilenceRunTokens;
  if (!strandedClick && !strandedWord) return null;
  const n = j + 1 + Math.min(r1, cfg.endedTailKeep);
  return n < tokens.length ? n : null;
}

/**
 * A terminal chunk that ended correctly and then free-ran an extra word. Returns
 * the token count to keep, or `null`.
 *
 * There is no silence seam here, so {@link isTrailingFiller} has nothing to
 * anchor on. Instead the earlier stop candidate must be strong, late and followed
 * by a short tail. The second acceptance path is narrower and exists for one
 * regression where the model never sampled a stop token but its best — very weak
 * — stop was 15 tokens before the hard ceiling.
 */
export function terminalEchoCut(
  tokenCount: number,
  eosPeakAt: number,
  eosPeakProb: number,
  minTokens: number,
  isTerminal: boolean,
  hitCeiling: boolean,
  cfg: PostprocessConfig
): number | null {
  if (!isTerminal) return null;
  if (!(eosPeakAt > Math.max(minTokens, 10) && eosPeakAt < tokenCount)) return null;

  const tail = tokenCount - eosPeakAt;
  const strongPeak =
    eosPeakProb >= cfg.echoStrongEosProbability &&
    tail <= cfg.echoStrongMaxTail &&
    eosPeakAt * 100 >= tokenCount * cfg.echoStrongMinPositionPct;
  const weakLatePeakAtCeiling =
    hitCeiling &&
    eosPeakProb >= cfg.echoWeakEosProbability &&
    tail <= cfg.echoWeakMaxTail &&
    eosPeakAt * 100 >= tokenCount * cfg.echoWeakMinPositionPct;
  return strongPeak || weakLatePeakAtCeiling ? eosPeakAt : null;
}

/**
 * Run every detector in precedence order and return one verdict.
 *
 * The shipped reader grew five entry points, one per field bug, and left the
 * ordering to each call site. Here they are one resolver with the precedence
 * written down, because an order that lives in a caller is an order the next
 * caller gets wrong.
 *
 * Peak-anchored rescues first, then the length-anchored one — it is the bluntest,
 * and it applies to *ended* rows too, because a model that babbles past its
 * sentence and only then samples a stop token has forfeited the trust that
 * stopping implies. The ended-tail trim runs only when nothing above fired.
 */
export function inspect(
  tokens: number[],
  req: InspectRequest,
  silence: Iterable<number>,
  cfg: PostprocessConfig
): Inspection {
  if (cfg.mode === "off" || tokens.length === 0) {
    return { keep: tokens.length, reason: "clean", suspect: false };
  }

  const sil = [...silence];
  let cut: number | null = null;
  let reason: Reason = "clean";

  // Terminal chunks only, like its three siblings. `isTerminal` means a
  // continuation chunk's stop peak is meaningless and its pauses are rhythm
  // rather than dead air — and this rule reads exactly those two signals, so it
  // was trimming mid-passage chunks on evidence the contract says is not
  // evidence. Changed in all five implementations together; postprocess is a
  // bit-parity surface.
  const fillerCut =
    req.isTerminal &&
    !req.ended &&
    req.eosPeakProb > cfg.fillerMinEosProbability &&
    req.eosPeakAt > Math.max(req.minTokens, 10) &&
    req.eosPeakAt < tokens.length &&
    isTrailingFiller(tokens, req.eosPeakAt, sil, cfg);

  // Early truncation first: nothing below can help a row that is already too
  // short, and the verdict is "incomplete" rather than "wrongly ended".
  if (isDropout(tokens.length, req.textTokenCount, cfg)) {
    return { keep: tokens.length, reason: "dropout", suspect: true };
  }

  // Then repetition, because it is the only rule that knows *exactly* where the
  // failure began. Every other anchor here is inferred from a signal that
  // might mean something else; an exactly repeated cycle is not.
  const looped = repetitionCut(tokens, silence, cfg);
  if (looped !== null) {
    cut = looped;
    reason = "repetition";
  } else if (fillerCut) {
    cut = req.eosPeakAt;
    reason = "silence_tail";
  } else {
    const echo = terminalEchoCut(
      tokens.length,
      req.eosPeakAt,
      req.eosPeakProb,
      req.minTokens,
      req.isTerminal,
      req.hitCeiling,
      cfg
    );
    if (echo !== null) {
      cut = echo;
      reason = "terminal_echo";
    } else {
      const desperate = desperationCut(
        tokens,
        req.textTokenCount,
        req.minTokens,
        req.eosPeakAt,
        sil,
        cfg,
        req.isTerminal
      );
      if (desperate !== null) {
        cut = desperate;
        reason = "desperation";
      }
    }
  }

  if (cut === null && req.ended) {
    const trimmed = endedTailTrim(tokens, sil, cfg, req.isTerminal);
    if (trimmed !== null) {
      cut = trimmed;
      reason = "ended_tail";
    }
  }

  // A condemned row that dodged every token anchor. Reported, never cut: no rule
  // could say where, and cutting at a guess is how the rescue truncated whole
  // sentences before the corroboration rules were added.
  const suspect =
    cut === null &&
    req.textTokenCount >= cfg.desperationMinTextTokens &&
    tokens.length >= req.textTokenCount * cfg.desperationSpeechPerTextToken;

  return { keep: cut ?? tokens.length, reason, suspect };
}
