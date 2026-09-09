/**
 * Deciding where a generated chunk actually ended.
 *
 * Mirrors `loudkit.postprocess`. This is a **detector**, not a filter: it reads
 * the speech tokens a chunk produced, answers one question (where did
 * the sentence really stop?) and returns a verdict. It never touches a sample of
 * audio.
 *
 * The artifact it removes is generated, not spectral. The decoder is
 * free-running, and silence tokens are exempt from the `min_p` cutoff (a pause
 * token is the only way to pause, and a filter that removes it removes
 * prosody), so once the sentence is over those tokens keep probability mass
 * indefinitely. The decoder free-runs silence, and any step where a
 * non-silence token survives the cutoff becomes a hallucinated word, heard
 * as "it finished, then a long gap, then one random word". Silence is not
 * exempt from the repetition penalty either: an exempt silence run is
 * absorbing mid-row as well; see {@link isStalled} for the failure that
 * guards against and the sampler for the measurement.
 *
 * Every constant came from a device trace or a regression, and every rule is
 * pinned by `tests/data/conformance/postprocess.json`, which all five ports run.
 * Provenance is in `docs/design/postprocess.md`.
 * Python reference: `loudkit/postprocess.py`.
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

/**
 * Which silence family the all-silence-cycle exemption in {@link repetitionCut}
 * reads. `acoustic` is the shipping default; `sampling` names the
 * pre-amendment law. See `PostprocessConfig.repetitionSilence`.
 */
export type RepetitionSilence = "acoustic" | "sampling";

export const REPETITION_SILENCE_VALUES: RepetitionSilence[] = ["acoustic", "sampling"];

/**
 * What a qualifying loop the decoder *resumed from* receives in
 * {@link inspect}. `condemn` is the shipping default; `cut` names the
 * pre-amendment law. See `PostprocessConfig.repetitionResume`.
 */
export type RepetitionResume = "condemn" | "cut";

export const REPETITION_RESUME_VALUES: RepetitionResume[] = ["condemn", "cut"];

/**
 * How many retry streams the seed ladder has room for: the gap between the
 * engine's retry stream (8) and its chunk streams (16). Stated here, beside
 * the attempt count it bounds, and read by the engine's ladder.
 */
export const RETRY_LADDER_HEADROOM = 8;

/** Which rule fired. `clean` means none did. */
export type Reason =
  | "clean"
  | "dropout"
  | "stall"
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
   * bestEOS=74@0.003 floor=31`: ~26 text tokens stopped only because it hit the
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
   * "It was as he expected." (14 text tokens) came back as 96 speech tokens of
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
  /**
   * A cap-hit row whose desperation cut keeps fewer speech tokens than this
   * many per text token is condemned into the retry ladder instead of
   * shipping the trim.
   *
   * The specimen: soren, da0028 chunk 4, seed 1234. The row burned 132 tokens
   * to the ceiling and the seam cut kept 36, 1.44 s of audio in which 33 of
   * the 36 kept tokens render near-silent through ids outside both manifest
   * censuses, so no set-membership rule can see them. The trim shipped a mute
   * chunk and the caller was never told to retry; the same window renders
   * clean at seeds 7 and 99.
   *
   * Calibrated on every cap-hit desperation rescue in the interior-stall and
   * acceptance batteries (nine rows, four voices, four languages): every keep
   * at or below 1.57 per text token was mute or missing much of its text, and
   * every keep at or above 1.85 carried real speech. 1.7 sits inside that
   * gap, and deliberately under 1.75, the floor of the measured healthy band
   * of speech tokens per text token, so a complete read is never condemned.
   * Cap-hit rows only: a row that ended on its own corroborated its trim
   * with a stop token. Zero disables the trigger.
   */
  desperationMinKeepPerTextToken: number;

  /** Silence before a blip that counts as stranding it (~0.24 s). */
  endedTailSilenceRun: number;
  /** <= 80 ms of "speech" is a click, not a word. */
  endedTailBlipMax: number;
  /**
   * A stray word behind a full seam on a *terminal* chunk is cut with it.
   * Continuation chunks keep their tails: their pauses are the sentence's
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
   * What a qualifying loop the decoder *resumed from* receives.
   *
   * The guard that holds without a census. A genuine lock-up is a tail
   * pathology: the model's own output is its context, the state is
   * absorbing, and the repeating region runs to the end of the row. Every
   * fire in the conformance fixture resumes by zero tokens, and a ceiling
   * can truncate at most one incomplete copy, `period - 1` tokens, at most
   * 11. A qualifying repetition followed by a full period or more of other
   * content is therefore a different event: a decoder that resumed was never
   * locked, and on a checkpoint without render censuses the thing it resumed
   * from is a pause parked on a silent-rendering id the sampler list cannot
   * name.
   *
   * `condemn` (the default): the row is reported whole (`keep` is the
   * full row, verdict repetition, `suspect`) and routed into the retry ladder
   * like `stall`. The cut is refused because the cut *is* the defect:
   * kathleen, en0023, seed 1234 on the published census-less pack parks a
   * pause on 6486 (x8) then 6405 (x24), the period-1 run passes every loop
   * condition, and the cut keeps 52 of 206 tokens, deleting the pause plus
   * the 131 tokens of correctly-read speech behind it: two sentences,
   * verdict repetition, no retry, audibly fluent. `repetitionSilence` closes
   * that on a manifest that carries the censuses; this field closes it on
   * every checkpoint, including ids no census lists.
   *
   * Not a discard-fraction guard, though one was calibrated first: a
   * fraction reads geometry, so a pause with less speech behind it slips
   * under any cap and ships the deletion as clean. Fires resume by 0 tokens,
   * the specimen by 131, and the largest resume a truncated genuine loop can
   * produce is `period - 1`, so the law is `resume >= period`:
   * integer-exact, no constant to tune, and a cut that survives it only ever
   * removes a tail, like every other rule in the layer.
   *
   * `cut` names the pre-amendment law, for a checkpoint measured under it.
   * The bare rule ({@link repetitionCut}) reports the loop either way; this
   * field decides what the resolver does with one that resumed.
   */
  repetitionResume: RepetitionResume;

  /**
   * Which silence family the all-silence-cycle exemption reads.
   *
   * `acoustic`: the union of the configured sampler silence ids and both
   * render censuses (`silenceRenderIds`, `quietRenderIds`). A pause parked on
   * *any* silent-rendering id is never mistaken for a decoder loop. The
   * specimen that settled it: kathleen, en0023, seed 1234. A mid-chunk pause
   * parked on ids 6486 (x7) then 6405 (x24), both rendering true silence,
   * both in the manifest's `silence_render_ids`, neither in the sampler's
   * `silence_token_ids`. Keyed to the sampler list, the exemption could not
   * see them: the 24-token period-1 run fired as a loop, and the cut kept one
   * cycle and deleted the pause plus the six seconds of correctly-read speech
   * behind it: two whole sentences, verdict repetition, not suspect, no
   * retry, audibly fluent. Six of the checkpoint's eight truly-silent ids sit
   * outside the sampler list, so this was the rule's default behaviour on
   * most real pauses; measured prevalence 1/200 renders, and the shape is
   * inaudible content loss shipping as clean.
   *
   * `sampling`: the configured sampler list alone, the pre-amendment law,
   * nameable so a checkpoint measured under it can declare what it measured.
   * A checkpoint without censuses gets this behaviour under either value,
   * since the union degenerates to the sampler list.
   *
   * This family feeds the loop exemption only. The tail rules
   * (`silence_tail`, `ended_tail`, the filler and desperation seams) stay
   * keyed to the sampler list they were calibrated against; see
   * `docs/design/postprocess.md` for the two-lists decision.
   */
  repetitionSilence: RepetitionSilence;

  /**
   * A non-tail dead-air run this long condemns the row (~1.0 s at 25 Hz).
   *
   * The run is measured two-class, and the two classes are essential: only
   * true-silence ids (`silenceRenderIds`) count toward this threshold, but the
   * run *continues* across quiet-family ids (`quietRenderIds`): breath and
   * decay tokens that render inaudible in context. Single-set counting was
   * measured broken: one breath token in the middle of real dead air split a
   * 47-token run into two short ones and the rule missed it.
   *
   * Calibrated across all ten shipping languages (120 passages per arm):
   * healthy interior runs top out at 13–19 tokens and healthy leading runs at
   * 11, so 25 is outside anything ordinary prose produced anywhere while
   * sitting under every measured stall. 20 also clears the healthy maxima; 25
   * is the shipped margin.
   */
  stallRunTokens: number;

  /**
   * Token ids that render as true digital silence.
   *
   * A property of the checkpoint, measured by rendering (per-id median energy
   * below -80 dBFS across two independent censuses), and therefore supplied by
   * the manifest (top level, beside `silence_token_ids`) precisely so the
   * next backend cannot re-guess it. Empty means the checkpoint predates the
   * census; the stall rule then runs its run trigger only, keyed to the
   * configured `silence_token_ids`, degraded (only 8 of that list's 31 ids
   * actually render silent, so the whole-row and majority triggers cannot be
   * trusted with it) but safe.
   *
   * NOT a sampling exemption list. Widening the sampler's `min_p` exemption to
   * exactly these ids was measured harmful (pause-time share doubles) and
   * the repetition penalty applies to every token regardless. This list exists
   * so the detectors read dead air where dead air actually is.
   */
  silenceRenderIds: number[];

  /**
   * The contextually-quiet family: breath and decay ids.
   *
   * Measured by per-instance RMS attribution (at least 90% of instances quiet,
   * 5+ sightings), minus the true-silence census. Dead-air runs continue
   * across these ids but they never count toward the run gate: a breath
   * inside dead air is still dead air, and a breath between words is not.
   * Manifest-supplied like `silenceRenderIds`; empty when the checkpoint
   * predates the census.
   */
  quietRenderIds: number[];

  /**
   * Early truncation: the row is too short to be the text it was asked for.
   * Reported, never cut: there is nothing to cut, and it is the most damaging
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
  desperationMinKeepPerTextToken: 1.7,
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
  repetitionResume: "condemn",
  repetitionSilence: "acoustic",
  stallRunTokens: 25,
  silenceRenderIds: [],
  quietRenderIds: [],
  dropoutMinTokens: 25,
};


/**
 * The reference's `PostprocessConfig._validate_ranges`, branch for branch.
 *
 * A constant out of range is not a preset, it is a typo in a manifest, and a
 * port that loads one runs detectors the reference refuses to build, so the
 * fingerprint agrees while the audio does not. The sentences are the
 * reference's word for word so a manifest refused by one engine is refused
 * with the same complaint by all five.
 *
 * The finite check `_validate_ranges` opens with has no branch here: the
 * manifest reader already refuses a non-finite number for any of these
 * fields, before a config exists to validate.
 */
export function validatePostprocessRanges(cfg: PostprocessConfig): void {
  if (!(cfg.retryMaxAttempts >= 0 && cfg.retryMaxAttempts < RETRY_LADDER_HEADROOM)) {
    throw new Error(
      `retry_max_attempts must be in [0, ${RETRY_LADDER_HEADROOM}): ` +
        `${cfg.retryMaxAttempts}. Above that the ladder's derived seeds run ` +
        "into the streams the chunk seeds use."
    );
  }
  if (cfg.repetitionMinCycles < 2) {
    // One cycle is not a repetition and two is the definition of one; a
    // threshold below two would cut every row that says a word twice.
    throw new Error(`repetition_min_cycles must be at least 2: ${cfg.repetitionMinCycles}`);
  }
  if (cfg.repetitionMaxPeriod < 1) {
    throw new Error(`repetition_max_period must be positive: ${cfg.repetitionMaxPeriod}`);
  }
  if (cfg.stallRunTokens < 1) {
    // At zero every row with a single silence token before speech is a stall,
    // and "condemned" stops meaning anything.
    throw new Error(`stall_run_tokens must be positive: ${cfg.stallRunTokens}`);
  }
  if (cfg.repetitionMinSpan < cfg.repetitionMinCycles) {
    // A span shorter than the cycle count is unreachable: the shortest
    // qualifying loop is min_cycles copies of a one-token cycle.
    throw new Error(
      `repetition_min_span (${cfg.repetitionMinSpan}) must be at least ` +
        `repetition_min_cycles (${cfg.repetitionMinCycles})`
    );
  }
  if (cfg.ceilingSpeechPerTextToken <= 0.0) {
    throw new Error(
      `ceiling_speech_per_text_token must be positive: ${cfg.ceilingSpeechPerTextToken}`
    );
  }
  if (cfg.desperationSpeechPerTextToken <= cfg.ceilingSpeechPerTextToken) {
    // The desperation rule exists for rows the ceiling let through. If it
    // triggered at or below the ceiling it would fire on every ceiling-stopped
    // row, including the ones the ceiling stopped correctly, and "certainly
    // broken" would stop meaning anything.
    throw new Error(
      `desperation_speech_per_text_token (${cfg.desperationSpeechPerTextToken}) ` +
        `must exceed ceiling_speech_per_text_token (${cfg.ceilingSpeechPerTextToken}): ` +
        "below it, the rule that means 'certainly broken' fires on rows the " +
        "ceiling stopped correctly"
    );
  }
  if (cfg.desperationMinKeepPerTextToken < 0.0) {
    throw new Error(
      `desperation_min_keep_per_text_token must be >= 0: ${cfg.desperationMinKeepPerTextToken}`
    );
  }
  if (cfg.desperationMinKeepPerTextToken > cfg.desperationBandRatio) {
    // The band top is where a real read could still have ended. Demanding a
    // keep above it condemns cuts landing exactly where the band admits them,
    // and "starved" stops meaning anything.
    throw new Error(
      `desperation_min_keep_per_text_token (${cfg.desperationMinKeepPerTextToken}) ` +
        `must not exceed desperation_band_ratio (${cfg.desperationBandRatio})`
    );
  }
  if (!(cfg.trailingFillerThreshold > 0.0 && cfg.trailingFillerThreshold <= 1.0)) {
    throw new Error(
      `trailing_filler_threshold must be in (0, 1]: ${cfg.trailingFillerThreshold}`
    );
  }
  if (!(cfg.fillerMinEosProbability >= 0.0 && cfg.fillerMinEosProbability < 1.0)) {
    throw new Error(
      `filler_min_eos_probability out of range: ${cfg.fillerMinEosProbability}`
    );
  }
  // The manifest spelling is the name the reference's message carries, so the
  // pairs are written out rather than derived from the camelCase keys.
  const nonNegative: [keyof PostprocessConfig, string][] = [
    ["ceilingSlackTokens", "ceiling_slack_tokens"],
    ["trailingSilenceRunTokens", "trailing_silence_run_tokens"],
    ["desperationBandFloor", "desperation_band_floor"],
    ["desperationMinTextTokens", "desperation_min_text_tokens"],
    ["endedTailSilenceRun", "ended_tail_silence_run"],
    ["endedTailBlipMax", "ended_tail_blip_max"],
    ["endedTailWordMax", "ended_tail_word_max"],
    ["fillerMaxSpeechAfterRun", "filler_max_speech_after_run"],
    ["endedTailKeep", "ended_tail_keep"],
    ["echoStrongMaxTail", "echo_strong_max_tail"],
    ["echoWeakMaxTail", "echo_weak_max_tail"],
    ["dropoutMinTokens", "dropout_min_tokens"],
  ];
  for (const [field, name] of nonNegative) {
    const value = cfg[field] as number;
    if (value < 0) throw new Error(`${name} must be >= 0: ${value}`);
  }
  const percentages: [keyof PostprocessConfig, string][] = [
    ["echoStrongMinPositionPct", "echo_strong_min_position_pct"],
    ["echoWeakMinPositionPct", "echo_weak_min_position_pct"],
  ];
  for (const [field, name] of percentages) {
    const pct = cfg[field] as number;
    if (!(pct >= 0 && pct <= 100)) throw new Error(`${name} is a percentage: ${pct}`);
  }
}

/** What the detectors concluded about one chunk. */
export interface Inspection {
  /**
   * How many leading tokens survive, equal to the input length when nothing
   * fired, so a caller can always slice by it without branching.
   */
  keep: number;
  reason: Reason;
  /**
   * The row is certainly wrong in a way no cut can fix. Set with `dropout`
   * (content missing), with `stall` (the row is dead air where speech should
   * be), with `repetition` on a loop the decoder resumed from (the cut would
   * delete what it came back to say, so the row is handed back whole), with
   * a starved `desperation` cut (a cap-hit trim that keeps less
   * than any full read of its text; here `keep` still holds the cut, as the
   * fallback if every retry is also condemned), and on a row impossibly long
   * for its text that dodged every token anchor. Not an error and not a cut:
   * a report, and the engine's signal to retry. Shipping such a row silently
   * is how the artifact reached listeners in the first place.
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
 * away: a model that stops on its own never reaches it.
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
 * Indices of chunks whose pace drifts past the tolerance from the median.
 *
 * Long-form drift: per-chunk pace (speech tokens over text tokens) against the
 * passage's own median, report-only. The median rather than the mean, so one
 * broken chunk cannot drag the baseline toward itself and hide.
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
 * The mechanism is the one behind the trailing hallucinated word (the
 * model's own output becomes its context), but it strikes mid-sequence, so no rule that
 * reads the end can find it.
 *
 * Deliberately hard to trigger, because it is the only rule here that anchors
 * mid-sequence: a short cycle, repeated many times, matched exactly. A decoder
 * that has genuinely locked up emits the same tokens rather than similar ones,
 * and a fuzzy match on a signal this destructive would truncate real speech.
 * And under `repetitionResume` `"condemn"` an *applied* cut only ever removes
 * a tail: a loop the decoder resumed from is condemned by the resolver
 * instead ({@link inspect} reads the resumption off `loopCandidate` and
 * judges it), so the mid-sequence anchor never deletes what followed it.
 *
 * A cycle that is entirely silence is never a loop: silence repeating is what
 * silence is, and the tail rules already judge pauses against where they sit.
 * Under `repetitionSilence` `"acoustic"` the exemption reads *acoustic*
 * silence (the passed ids unioned with both render censuses) the same way
 * {@link isStalled} reads its censuses off the config. Keyed to the sampler
 * list alone it was blind to six of the eight truly-silent ids, and a long
 * pause parked on one of them fired as a period-1 loop whose cut deleted the
 * pause and every correctly-read token behind it (kathleen, en0023, seed
 * 1234: two sentences, verdict repetition, audibly fluent). A cycle mixing
 * silence with speech still counts: a word-then-pause stutter is one of the
 * shapes this failure takes.
 *
 * Returns one full cycle past the loop's start: the first instance is plausibly
 * the word the sentence wanted.
 */
export function repetitionCut(
  tokens: readonly number[],
  silence: Iterable<number>,
  cfg: PostprocessConfig
): number | null {
  return loopCandidate(tokens, silence, cfg)?.cut ?? null;
}

/**
 * The earliest qualifying loop: `{cut, resumed}`.
 *
 * One search serves both questions. The cut index is {@link repetitionCut}'s
 * contract, unchanged. `resumed` is whether the winning loop's repeating
 * region ends `period` or more tokens before the row does: a locked decoder
 * emits its cycle to the end of the row, and a ceiling can truncate at most
 * one incomplete copy (`period - 1` tokens), so a full period of anything
 * else after the region means the decoder came back, which a locked decoder,
 * by definition, does not. No extra scan pays for it: a matching full copy
 * would have been counted as another cycle, so `n - at >= period` already
 * implies a deviation.
 */
function loopCandidate(
  tokens: readonly number[],
  silence: Iterable<number>,
  cfg: PostprocessConfig
): { cut: number; resumed: boolean } | null {
  const n = tokens.length;
  if (n < cfg.repetitionMinSpan) return null;
  // The exemption's family, not the run rules': the tail rules keep reading
  // the sampler list they were calibrated against. Resolved here rather than
  // by the caller for the same reason `isStalled` reads its censuses off the
  // config: a family that lives in a caller is a family the next caller
  // feeds wrong. Without censuses the union is the sampler list, unchanged.
  const quietIds =
    cfg.repetitionSilence === "acoustic"
      ? new Set([...silence, ...cfg.silenceRenderIds, ...cfg.quietRenderIds])
      : new Set(silence);
  const quiet = tokens.map((t) => quietIds.has(t));

  // Earliest loop wins: a row that locks up twice locked up first at the first
  // one, and everything after it is already inside the failure.
  let best: { cut: number; resumed: boolean } | null = null;
  const longestPeriod = Math.min(cfg.repetitionMaxPeriod, Math.floor(n / cfg.repetitionMinCycles));
  for (let period = 1; period <= longestPeriod; period++) {
    for (let start = 0; start + period * cfg.repetitionMinCycles <= n; start++) {
      let cycles = 1;
      let at = start + period;
      for (; at + period <= n; at += period) {
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
        if (best === null || candidate < best.cut) {
          best = { cut: candidate, resumed: n - at >= period };
        }
        break;
      }
    }
  }
  return best;
}

/**
 * Whether the decoder spent this row trapped in silence.
 *
 * The failure the tail rules structurally cannot see. The decoder enters a
 * silence run at a pause point (its own argmax) and, with `min_p` stripping
 * every non-silence candidate while the exemption re-admits the listed silence
 * ids, the run's exit probability is effectively zero (measured: zero escapes
 * in 1,031 instrumented trap steps). The sampler now applies the repetition
 * penalty to silence, which closes the trap at the source; this rule is the
 * detector for what still gets through, and for any checkpoint or
 * configuration where the trap re-opens. At scale before the fix: 33.0% of
 * paragraphs carried a >1 s hole and 74 chunks in 1705 passages rendered no
 * speech at all (11.5 minutes of text silently swallowed), every one of them
 * `clean`, because all six other rules anchor on the tail.
 *
 * Three triggers, all integer-exact, any one condemns:
 *
 * - **no speech at all**: every generated token is in the silence-or-quiet
 *   family. Measured: mute rows are 255/255 silence tokens, a seed lottery
 *   (they recur at 3/18 re-renders), and a retry rescues 9/10.
 * - **a non-tail dead-air run** of at least `stallRunTokens` true-silence
 *   tokens. Two-class: quiet-family ids extend a run without counting toward
 *   it (see `stallRunTokens` for why single-set counting is broken). Tail runs
 *   are excluded: the tail rules own the tail, and a trailing pause is judged
 *   against the place it sits in.
 * - **a ceiling overrun that is mostly silence**: `hitCeiling` and the family
 *   holds a strict majority of the row. A tail run on a cap-hit row is not a
 *   natural tail: the ceiling truncated the read, so the dead air is a stall
 *   the cap happened to interrupt (the reference specimen is 90.2% silence, a
 *   210-token run to the cap). This also closes a structural hole: the ceiling
 *   clips rows to 4.0x text tokens + 40, so past 80 text tokens a cap-hit
 *   stall can never reach the 4.5x desperation threshold: the rule that means
 *   "certainly broken" was unreachable by the most broken rows this layer
 *   sees.
 *
 * Without a census (`silenceRenderIds` empty, a checkpoint packed before it)
 * only the run trigger fires, keyed to the configured `silence_token_ids`.
 * That is the measured-safe subset: 13 of the configured list's 31 ids render
 * audible speech, so whole-row membership in that list does not prove a mute
 * row, and a whole-row trigger keyed to it could condemn real speech.
 * Degraded-but-safe beats a fallback that lies.
 *
 * Returns `true` for a condemned row. There is nothing to cut: the failure is
 * a hole, not a tail, and the fix is the retry ladder, the same route
 * `dropout` takes, for the same reason.
 */
export function isStalled(
  tokens: readonly number[],
  hitCeiling: boolean,
  silence: Iterable<number>,
  cfg: PostprocessConfig
): boolean {
  if (tokens.length === 0) return false;
  const census = cfg.silenceRenderIds.length > 0;
  const gate = new Set(census ? cfg.silenceRenderIds : silence);
  const family = new Set(gate);
  for (const id of cfg.quietRenderIds) family.add(id);

  const inFamily = tokens.map((t) => family.has(t));
  const familyCount = inFamily.filter(Boolean).length;
  if (census && familyCount === tokens.length) return true;
  if (census && hitCeiling && 2 * familyCount > tokens.length) return true;

  let gateCount = 0;
  for (let i = 0; i < tokens.length; i += 1) {
    if (inFamily[i]) {
      if (gate.has(tokens[i])) gateCount += 1;
    } else {
      // The run ended before the row did, so it is not the tail.
      if (gateCount >= cfg.stallRunTokens) return true;
      gateCount = 0;
    }
  }
  return false;
}

/**
 * Whether what follows `index` is a trailing tail rather than more sentence.
 *
 * The overrun rescue cuts back to where the model came closest to stopping, and
 * that peak is a hint, not a verdict. Trusting it alone truncated whole
 * sentences: a voice reading a language its tag does not match may never commit
 * to stopping, so its best moment of hesitation lands a third of the way in.
 *
 * So the peak is corroborated by *what it proposes to discard*: either the tail
 * is mostly silence by share, or it holds a long unbroken run with only a stray
 * word behind it. Without that second half, a rhetorical pause mid-tail (25
 * silent tokens, then 80 of speech) matched the run rule and the rescue cut the
 * rest of the sentence off.
 */
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

  // Collect qualifying runs, then require every gap of speech between them
  // (and after the last) to be a stray word or less. [seam][real
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

  // No seam: the babble is dense; fall back to the model's own best stop, if
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
 * regression where the model never sampled a stop token but its best stop, very
 * weak, was 15 tokens before the hard ceiling.
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
 * The order, which is the contract:
 *
 * 1. `dropout`: the row is too short for the text. Reported whole, never
 *    cut: nothing below can help a row that is missing content.
 * 2. `repetition`: an exact repeated cycle. First of the cuts, because it is
 *    the only rule that knows exactly where the failure began; every other
 *    anchor here is inferred. A cycle the decoder came back from is condemned
 *    whole rather than cut, since the cut would delete what it came back to say.
 * 3. `stall`: a mid-row hole. Condemned whole, before any tail rescue: a tail
 *    cut cannot remove a hole in the middle, and a rescue firing here would
 *    trim the tail and ship the hole under its own reason.
 * 4. `silence_tail`: the peak-anchored filler trim.
 * 5. `terminal_echo`, then `desperation`: the length-anchored one is the
 *    bluntest, and it applies to *ended* rows too, because a model that babbles
 *    past its sentence and only then samples a stop token has forfeited the
 *    trust that stopping implies.
 * 6. `ended_tail_trim`: only when nothing above fired.
 *
 * The order above is the order the code applies, and `repetition` and `stall`
 * come before the peak-anchored rescues because neither is peak-anchored: a
 * repeated cycle is evidence on its own. All five implementations run it in
 * this order.
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
  let starved = false;

  // Terminal chunks only, like its three siblings. `isTerminal` means a
  // continuation chunk's stop peak is meaningless and its pauses are rhythm
  // rather than dead air, and this rule reads exactly those two signals, so it
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
  const looped = loopCandidate(tokens, silence, cfg);
  if (looped !== null && looped.resumed && cfg.repetitionResume === "condemn") {
    // The decoder came back after the repeating region, so it was never
    // locked, and the cut would delete whatever it came back to say: the
    // en0023 defect exactly, on any checkpoint whose silence family cannot
    // name the pause the region actually was. Condemned like `stall`, whole:
    // unlike a starved desperation cut there is no trim worth keeping as a
    // fallback, because the trim is the defect.
    return { keep: tokens.length, reason: "repetition", suspect: true };
  }
  if (looped !== null) {
    cut = looped.cut;
    reason = "repetition";
  } else if (isStalled(tokens, req.hitCeiling, sil, cfg)) {
    // Condemned, never cut, before any tail rescue can run: a mid-row hole is
    // not removable by a tail cut, and a rescue that fired here would trim the
    // tail and ship the hole under its own reason. Routed like `dropout`:
    // reported whole, suspect, into the retry ladder.
    return { keep: tokens.length, reason: "stall", suspect: true };
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
        // The starved rescue. On a cap-hit row the trim has no stop token
        // corroborating it, and a cut keeping fewer than
        // `desperationMinKeepPerTextToken` speech tokens per text token kept
        // less than any full read of the text. The kept audio can be
        // near-silence through ids no census lists (da0028: 33 of the 36 kept
        // tokens, a mute chunk shipped as fixed), so the keep's *length* is
        // the only evidence there is. Condemned like `stall`, but the cut
        // stands: if the retry ladder exhausts, the trim ships (today's
        // audio) flagged `suspect`, rather than the untrimmed babble.
        starved =
          req.hitCeiling && desperate < req.textTokenCount * cfg.desperationMinKeepPerTextToken;
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
    starved ||
    (cut === null &&
      req.textTokenCount >= cfg.desperationMinTextTokens &&
      tokens.length >= req.textTokenCount * cfg.desperationSpeechPerTextToken);

  return { keep: cut ?? tokens.length, reason, suspect };
}
