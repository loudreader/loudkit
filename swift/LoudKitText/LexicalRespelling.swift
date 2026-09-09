import Foundation

/// English words inside Polish text, respelled the way a Polish reader says
/// them.
///
/// The engine is grapheme-based with ONE language tag per utterance, so a
/// Polish render reads "download" with Polish letter-to-sound rules and
/// mangles it. Three fixes were auditioned on the same sentence: inline
/// `[en]...[pl]` tag switching, phonetic respelling, and nothing. Respelling
/// won by ear, which makes sense: "dałnloud" is not a hack, it is how the word
/// actually sounds in a Polish sentence, accent included. A Pole saying
/// "download" says a Polish-phonology version of it; the lexicon writes that
/// down.
///
/// Dictionary-first and ONLY dictionary: a rule-based English G2P bolted on
/// here would misfire on real Polish words and the cost of a false positive
/// (mangling native text) is far higher than the cost of a miss (one
/// anglicism read wrong, which is where we started). Inflections ride as
/// suffixes: Poles decline these words ("maila", "deadline'u"), so matching
/// is stem + known Polish ending, with the apostrophe forms handled.
public enum LexicalRespelling {

    /// Respell `text` for the given language. Only Polish has a lexicon today;
    /// every other language returns the text untouched.
    ///
    /// Case-insensitive on the language id: the tokenizer lowercases its tag,
    /// so a caller passing `"PL"` would otherwise get Polish tokenisation
    /// without Polish respelling.
    static func applied(to text: String, languageId: String?) -> String {
        guard languageId?.lowercased() == "pl" else { return text }
        return respellWords(in: respellPhrases(in: respellSymbols(in: text)))
    }

    /// Math and unit symbols the model cannot say, as Polish words, with
    /// context guards, because "-" is also a hyphen and "/" is also a path.
    /// Everything here fires only where digits or spacing make the meaning
    /// unambiguous; the ambiguous cases stay untouched rather than guessed.
    static func respellSymbols(in text: String) -> String {
        var out = text
        let rules: [(String, String)] = [
            (#"(?<=\d)\s?%"#, " procent"),
            (#"(?<=\d)\s?°C"#, " stopni Celsjusza"),
            (#"(?<=\d)\s?°"#, " stopni"),
            (#"(?<=\d)\s*/\s*(?=\d)"#, " przez "),
            (#"(?<=\d)\s*\*\s*(?=\d)"#, " razy "),
            (#"(?<=\d)\s*\^\s*(?=\d)"#, " do potęgi "),
            (#"\s=\s"#, " równa się "),
            (#"\s\+\s"#, " plus "),
            (#"\s<\s"#, " mniejsze niż "),
            (#"\s>\s"#, " większe niż "),
            (#"\s-\s"#, " minus "),
        ]
        for (pattern, replacement) in rules {
            out = out.replacingOccurrences(of: pattern, with: replacement,
                                           options: .regularExpression)
        }
        return out
    }

    // MARK: the curated tables

    /// `pl_respell_rules.json`, parsed and shape-checked once.
    ///
    /// The hand-written half of Polish respelling: the phrases, the curated
    /// lexicon, and the three word lists the inflection walk consults. Data
    /// rather than literals for the same reason the number grammars are, so
    /// that five implementations read one file and a word approved by ear is
    /// approved once. Mirrors `loudkit.frontend.speechtext._PolishRules`.
    /// `keep_polish` and `function_words` arrive through
    /// `convertFromSnakeCase` below rather than a `CodingKeys` table, which
    /// keeps the file's names and this port's names visibly the same pair.
    private struct Rules: Decodable {
        let phrases: [[String]]
        let lexicon: [String: String]
        let keepPolish: [String]
        let functionWords: [String]
        let endings: [String]
    }

    /// A phrase entry is written form and spoken form, and nothing else.
    private static let phrasePair = 2

    /// The file, read once, lazily, through the resolver every pass that reads
    /// a bundled table uses.
    ///
    /// A trap rather than a fallback, which is where this file differs from
    /// `pl_en_respell.json` next door. That one degrades to an empty lexicon
    /// and says so in a log, because it is inside `grammar_digest`: a build
    /// that cannot read it computes a different digest and an engine pairing it
    /// with any other refuses to start. This file sits *below* the digest, so
    /// an absent or malformed copy would move the spoken words with the
    /// fingerprint unchanged and nothing anywhere to report it. Python raises
    /// at the same three checks for the same reason.
    private static let rules: Rules = {
        guard let data = Numbers.resourceBytes("pl_respell_rules") else {
            preconditionFailure(
                "pl_respell_rules.json is not in the bundle. It ships as a resource of "
                    + "LoudKitText; copy it with tools/sync_grammar.py.")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let decoded = try? decoder.decode(Rules.self, from: data) else {
            preconditionFailure("pl_respell_rules.json is not the object this build reads.")
        }
        guard decoded.phrases.allSatisfy({ $0.count == phrasePair }) else {
            preconditionFailure(
                "pl_respell_rules.json: each phrase must be [written, spoken].")
        }
        return decoded
    }()

    /// Multi-word anglicisms respelled as a unit, BEFORE the word pass, the
    /// existence proof is "release notes": word by word, "notes" is a Polish
    /// homograph (the notebook) and must stay Polish, but inside this phrase
    /// it is unmistakably English. The phrase pass sees the context a word
    /// pass cannot.
    ///
    /// Ordered, and applied in the file's order.
    static let phrases: [(String, String)] = rules.phrases.map { ($0[0], $0[1]) }

    /// English words that are ALSO everyday Polish words and slipped past the
    /// frequency gate: the word pass leaves them alone, and only a phrase
    /// above may respell them.
    /// Two families: Polish homographs, and loanwords Poles read
    /// ORTHOGRAPHICALLY, "bug" is [bug] in Polish mouths, never [bag], so
    /// the English respelling would be the mangling.
    static let keepPolish: Set<String> = Set(rules.keepPolish)

    private static func respellPhrases(in text: String) -> String {
        var out = text
        for (phrase, spoken) in phrases {
            out = out.replacingOccurrences(of: phrase, with: spoken,
                                           options: [.caseInsensitive])
        }
        return out
    }

    /// `\p{Nd}` for a `Character`, which is what `str.isdecimal()` means.
    ///
    /// `Character.isNumber` is Numeric_Type and answers true for `\u{00b2}`, `\u{2462}` and
    /// the CJK numerals; every one of those reached a number path here that
    /// Python's `isdecimal()` keeps them out of.
    static func isDecimal(_ ch: Character) -> Bool {
        guard ch.unicodeScalars.count == 1, let sc = ch.unicodeScalars.first else {
            return false
        }
        return SpeechText.isDecimalDigit(sc)
    }

    private static func respellWords(in text: String) -> String {
        // words[i] with seps[i+1] after it; seps[0] is anything before the
        // first word. Clean alternation, the previous interleaved collector
        // mis-aligned the classification array and dropped words out of spans.
        var words: [String] = []
        var seps: [String] = [""]
        var inWord = false
        for ch in text {
            // The apostrophe stays inside the word: "deadline'u" is one token
            // to a Polish reader and its ending must survive the respelling.
            if ch.isLetter || ch.isNumber || ch == "'" || ch == "’" {
                if !inWord { words.append(""); inWord = true }
                words[words.count - 1].append(ch)
            } else {
                if inWord { seps.append(""); inWord = false }
                seps[seps.count - 1].append(ch)
            }
        }
        if inWord { seps.append("") }

        // A RUN of English words is a quotation, not code-switching: four or
        // more in a row read better with the real [en] rules than as four
        // Polish transliterations in a trenchcoat. Short bursts stay with the
        // lexicon, the ear test picked transliteration for those.
        let isEnglish = words.map { word -> Bool in
            let lower = word.lowercased()
            return spelledAcronym(word) == nil && !keepPolish.contains(lower)
                && !polishFunctionWords.contains(lower)
                && (lookup(lower) != nil || isEnglishWord(lower))
        }
        var out = seps[0]
        var i = 0
        // `isdecimal()`, the way Python asks it: a `No`, an `Nl` or a numeric
        // `Lo` is not a digit group.
        func isDigits(_ w: String) -> Bool {
            !w.isEmpty && w.allSatisfy(isDecimal)
        }
        while i < words.count {
            // The whole run of digit groups is measured before any of it is
            // read, because the decision belongs to the run and not to its
            // first pair. Two groups is a decimal, "dwa przecinek pięć".
            // Three or more is a version, an address or a date, and is left
            // exactly as written.
            //
            // Reading only the first pair turns "192.168.0.1" into "sto
            // dziewięćdziesiąt dwa przecinek jeden sześć osiem" with ".0.1"
            // trailing behind it, and skipping just that pair moves the same
            // mistake one group along; `numbers.expand` refuses these by the same
            // rule, and this reader has to agree with it.
            //
            // The same run when it starts with a token that has a letter in
            // it: `v1.2.3` collects as ["v1", "2", "3"] and the branch below
            // only starts on a pure-digit group, so the chain is never
            // measured and the version comes out "fał jeden.dwa przecinek
            // trzy".
            if Self.spelledCodeToken(words[i]) != nil {
                var end = i
                while end + 1 < words.count,
                      seps[end + 1] == "." || seps[end + 1] == ",",
                      isDigits(words[end + 1]) || Self.spelledCodeToken(words[end + 1]) != nil {
                    end += 1
                }
                if end > i {
                    for k in i...end { out += words[k] + seps[k + 1] }
                    i = end + 1
                    continue
                }
            }

            if isDigits(words[i]) {
                var end = i
                while end + 1 < words.count, isDigits(words[end + 1]),
                      seps[end + 1] == "." || seps[end + 1] == "," {
                    end += 1
                }
                let groups = end - i + 1
                if groups >= 3 {
                    for k in i...end { out += words[k] + seps[k + 1] }
                    i = end + 1
                    continue
                }
                if groups == 2 {
                    let whole = Self.numberWords(words[i]) ?? words[i]
                    // `map` with a fallback, not `compactMap`: the gate above is
                    // `\p{Nd}` and this table is ASCII, so a digit it cannot name
                    // would be dropped from the utterance rather than read badly.
                    // `respelled` states the rule for the same table below.
                    let frac = words[i + 1].map { digitWords[$0] ?? String($0) }
                        .joined(separator: " ")
                    out += whole + " przecinek " + frac + seps[i + 2]
                    i += 2
                    continue
                }
            }
            if isEnglish[i] {
                var j = i
                while j < words.count, isEnglish[j] { j += 1 }
                if j - i >= 4 {
                    // Inside a detected English span every word transliterates,
                    // gate ignored, "brown" alone stays Polish, "brown" inside
                    // "the quick brown fox" becomes "brałn". The inline-[en]-tag
                    // version of this block lost the ear test decisively: the
                    // model's prosody falls apart on mid-sentence tag switches.
                    for k in i..<j {
                        let lower = words[k].lowercased()
                        let hit = lexicon[lower] ?? payload.all[lower]
                        out += matchCase(of: words[k], onto: hit ?? words[k]) + seps[k + 1]
                    }
                    i = j
                    continue
                }
            }
            out += respelled(words[i]) + seps[i + 1]
            i += 1
        }
        return out
    }

    // MARK: acronyms

    /// How long an all-caps token may be before this pass stops calling it an
    /// acronym.
    ///
    /// `Letters.spellAcronym` reads a listed word acronym of any length as a
    /// word, which is the right answer where it is asked, at the top of the
    /// funnel with the neighbouring capitals still visible. Down here the only
    /// question is whether the English-run detector should skip the token, and
    /// a long all-caps run is a heading or a shout far more often than an
    /// initialism.
    private static let maxAcronym = 5

    /// The acronym reading the English-run gate above asks about.
    ///
    /// `Letters` owns the spelling for all twelve languages, GPT → "gie-pe-te"
    /// out of the grammar file's own table, with the allowlist of acronyms said
    /// as WORDS (NASA, RAM, PIN) beside it; this adds the length cap that makes
    /// the question a narrower one.
    private static func spelledAcronym(_ word: String) -> String? {
        guard word.count <= maxAcronym else { return nil }
        return Letters.spellAcronym(word, language: "pl")
    }

    // MARK: the generated long tail

    /// CMUdict → Polish orthography, ~110k words, generated by
    /// `tools/gen_pl_respell.py` with the common-Polish gate baked in.
    /// The curated lexicon above always wins, its forms were approved by ear.
    ///
    /// `ChatterboxAssets` first, so an application shipping its own copy still
    /// overrides this one, then `Bundle.module`, where the packaged resource
    /// lives, the same way `Numbers.swift` next door reads `numbers.json`.
    ///
    /// Both channels rather than `ChatterboxAssets` alone: one asset channel
    /// is one set of failure modes, and the cost of the single channel is worse
    /// than a flake. `swift test` does not populate the asset channel, so with
    /// that channel alone the lexicon is simply absent there: the funnel's
    /// last pass switches itself off, says so in a log nobody reads, and every
    /// test in the package goes on passing, the conformance fixture included,
    /// because it carries no respelling case that would notice.
    private struct Payload: Decodable {
        let respell: [String: String]
        let words: [String]
        let respellAll: [String: String]
        let polish: [String]
    }

    private static let payload: (respell: [String: String], words: Set<String>,
                                 all: [String: String], polish: Set<String>) = {
        // The same resolver `Numbers.grammarDigest` hashes through, so the
        // fingerprint always describes the file this pass actually read. If
        // they differed, this pass preferring `ChatterboxAssets`, the digest
        // reading `Bundle.module` unconditionally, an application shipping its
        // own lexicon would speak from one file and report the digest of
        // another.
        guard let data = Numbers.resourceBytes("pl_en_respell"),
              let decoded = try? JSONDecoder().decode(Payload.self, from: data) else {
            // An empty lexicon is not a degraded mode, it is a different funnel:
            // the last pass stops running and the package speaks the English
            // inside Polish text with Polish letter values. Said at error level
            // *and* visible in the fingerprint, because a log line nobody reads
            // is no safeguard, `grammarDigest`
            // resolves the same way and returns "" when the file is missing, so
            // an engine pairing this build with any other refuses to start
            // rather than speaking differently in silence.
            Log.error("pl_en_respell.json missing: long-tail respelling disabled")
            return ([:], [], [:], [])
        }
        return (decoded.respell, Set(decoded.words), decoded.respellAll,
                Set(decoded.polish))
    }()

    private static var generated: [String: String] { payload.respell }

    /// "Is this an English word at all", the span detector's question, and
    /// deliberately UNGATED: gating dropped "brown" and "dog" (subtitle leak
    /// into the Polish list) and broke the spans they sat inside.
    static func isEnglishWord(_ lower: String) -> Bool { payload.words.contains(lower) }

    /// Polish function words that happen to spell English words ("i" = I,
    /// "to" = to, "on" = on), never span members, or a span eats the Polish
    /// conjunction after it.
    static let polishFunctionWords: Set<String> = Set(rules.functionWords)

    private static func lookup(_ word: String) -> String? {
        if let curated = lexicon[word] { return curated }
        if keepPolish.contains(word) { return nil }
        return generated[word]
    }

    /// How many digits a bare token may carry and still be read as one number.
    ///
    /// Past six the run is an identifier, an order number or a phone number far
    /// more often than a quantity, and *dziewięćset osiemdziesiąt siedem
    /// tysięcy* is a worse reading of `9876543210` than the digits themselves.
    private static let maxSpokenDigits = 6

    /// A run of digits as Polish cardinal words, or `nil` to leave it alone.
    ///
    /// The grammar is `Numbers.cardinal`'s, base (masculine) forms, "sto
    /// dwadzieścia trzy", deliberately NOT inflected against the following noun
    /// (that needs its gender and case); the occasional "dwa gruszki" still
    /// beats the model free-styling over raw digits, which the ear test rated
    /// unprintably. What this adds is the two refusals the respelling pass
    /// makes on top of it, a leading zero and a run too long to be a quantity,
    /// both of which read better digit by digit.
    static func numberWords(_ token: String) -> String? {
        // Unicode decimal digits are digits. `Int("١٢٣")` is nil in Swift,
        // and the caller's fallback then mapped every character through an
        // ASCII-keyed table with `compactMap`, which drops what it cannot
        // map, so a Polish passage containing Arabic-Indic numerals lost them
        // entirely and read as if they had never been written. Python reads
        // the same string as "sto dwadzieścia trzy"; normalising here is what
        // makes the two agree.
        //
        // `\p{Nd}` and not `Character.isNumber`, which is Numeric_Type: that
        // reads `\u{4e00}\u{4e8c}\u{4e09}` -- three CJK ideographs, category `Lo`, numeric
        // values 1, 2, 3 -- as `123`, so Swift alone said *sto dwadzieścia
        // trzy* for a word Python leaves written. Python asks `isdecimal()`
        // here for the same reason `int()` does.
        let ascii = String(token.map { ch -> Character in
            guard ch.unicodeScalars.count == 1, let sc = ch.unicodeScalars.first,
                  SpeechText.isDecimalDigit(sc),
                  let value = ch.wholeNumberValue, (0...9).contains(value)
            else { return ch }
            return Character(String(value))
        })
        // The leading-zero test reads the ORIGINAL token, not the normalised
        // one. Python asks `token.startswith("0")` against the ASCII zero, so a
        // fullwidth "０１２３" is not a leading-zero number to it and reads as
        // one hundred and twenty three; Go, Rust and JS agree. Normalising
        // first turned it into "0123", which this guard then refused, and
        // Swift alone spelled the four glyphs one by one.
        guard ascii.count <= maxSpokenDigits, !token.hasPrefix("0") || token == "0",
              let value = Int64(ascii) else { return nil }
        return Numbers.cardinal(value, language: "pl")
    }

    /// The ten ASCII digits as Polish words, for reading a token one character
    /// at a time.
    ///
    /// ASCII only, and a table rather than ten `cardinal` calls per token,
    /// because the callers below ask "is this a character I can say" of every
    /// character of every word. `\p{Nd}` is true of `٥` and `൬` as well, and a
    /// token holding one is not a token this pass says character by character.
    private static let digitWords: [Character: String] = {
        var table: [Character: String] = [:]
        for digit in 0...9 {
            guard let word = Numbers.cardinal(Int64(digit), language: "pl") else { continue }
            table[Character(String(digit))] = word
        }
        return table
    }()

    /// The letter names a mixed letter-digit token is spelled with.
    ///
    /// The names are the grammar file's, so `GPT` is *gie-pe-te* in one place.
    /// The ASCII narrowing is this pass's own: `numbers.json` also names the
    /// nine Polish letters, and reading them here would turn `Ż1`, left written
    /// today, into *żet jeden*. That is a change in what Polish says, so it
    /// belongs to a `FUNNEL_PORTED` bump and not to a table move.
    private static let codeLetterNames: [Character: String] = {
        var table: [Character: String] = [:]
        for ch in "abcdefghijklmnopqrstuvwxyz" {
            guard let name = Letters.letterName(String(ch), language: "pl") else { continue }
            table[ch] = name
        }
        return table
    }()

    /// Code tokens, hashes, "utf8", "T986", are neither Polish nor English:
    /// they are SPELLED, letter names in Polish, digits as words. Capped, so
    /// a 40-char commit hash does not become forty words of noise.
    /// Spelling `R2` character by character is how a Polish reader says it;
    /// doing the same to an eleven-character identifier is a wall of letter
    /// names nobody follows. Past this the token is left written.
    private static let maxSpelledCode = 8

    private static func spelledCodeToken(_ word: String) -> String? {
        let hasLetter = word.contains { $0.isLetter }
        let hasDigit = word.contains(where: isDecimal)
        guard hasLetter, hasDigit else { return nil }
        // All or nothing. A character with no letter name was skipped in
        // silence, so `Müller123` came out *em el el e er jeden dwa*, the `ü`
        // simply gone, a name changed rather than mispronounced; and the
        // eight-character cap truncated instead of refusing, so `żelazny2024`
        // lost its last digits. A token half-read is worse than one left
        // written, because the listener cannot tell anything was dropped.
        guard word.count <= maxSpelledCode else { return nil }
        var parts: [String] = []
        for ch in word {
            if let digit = digitWords[ch] {
                parts.append(digit)
                continue
            }
            guard let name = codeLetterNames[Character(ch.lowercased())] else { return nil }
            parts.append(name)
        }
        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: " ")
    }

    private static func respelled(_ word: String) -> String {
        // No acronym branch here. `Letters.applied(to:language:)` owns that
        // decision for all twelve languages and takes it earlier in the funnel,
        // where the surrounding capitals are still visible; this pass sees one
        // word at a time and so cannot tell an initialism from a shout. Deciding
        // it here spells "THIS IS FINE" as te-ha-i-es i-es ef-i-en-e, and "CIA
        // CIA" as ce-i-a ce-i-a where the earlier pass has already decided that
        // a run of capitals is emphasis. `spelledAcronym` stays: the English-run
        // test above asks it whether a word is an acronym, which is a different
        // question from spelling one.
        if let code = spelledCodeToken(word) { return code }
        let lower = word.lowercased()
        if let hit = lookup(lower) { return matchCase(of: word, onto: hit) }
        // Digits-only tokens: cardinal words when sane, digit-by-digit when
        // weird (leading zeros, longer than six digits). Raw digits reached
        // the model exactly once and the ear test rated them unprintably.
        if !word.contains(where: { $0.isLetter }) {
            if let cardinal = Self.numberWords(word) { return cardinal }
            // Digit by digit, but never to *nothing*. `compactMap` silently
            // dropped every character the ASCII-keyed table did not know, so
            // an unmappable token was deleted from the utterance rather than
            // read badly. A word this function cannot improve is returned
            // unchanged; losing text is not an available outcome.
            let spelled = word.map { digitWords[$0] ?? String($0) }
            return spelled.joined(separator: " ")
        }
        // Nothing under three letters declines from a dictionary stem, and
        // Polish is full of one-letter words ("i", "w", "z") that made the
        // stem range below crash outright.
        guard lower.count > 3 else { return word }

        // A word the Polish frequency list knows is POLISH: hands off. The
        // stem walk below once matched "temperatura" as temperature+a and
        // produced "tempraczera", an English stem plus a Polish ending is
        // only evidence of borrowing when the whole word is not already
        // ordinary Polish.
        if payload.polish.contains(lower) { return word }
        // Inflected: longest dictionary stem + a known Polish ending, with or
        // without the apostrophe ("deadline'u", "maila", "updatem").
        for cut in (2..<lower.count).reversed() {
            let idx = lower.index(lower.startIndex, offsetBy: cut)
            let stem = String(lower[..<idx])
            var suffix = String(lower[idx...])
            if suffix.first == "'" || suffix.first == "’" { suffix.removeFirst() }
            // Silent-e stems: "update" declines as "updatem", stem loses the e.
            var hit = lookup(stem)
            if hit == nil { hit = lookup(stem + "e") }
            guard var base = hit, polishEndings.contains(suffix) else { continue }
            // The respelling's trailing vowel folds into a vowel-initial ending
            // ("dedlajn" + "u", but "miting" + "u", only vowels collide).
            if let lastVowel = base.last, "aeiouy".contains(lastVowel),
               let first = suffix.first, "aeiouy".contains(first) {
                base.removeLast()
            }
            return matchCase(of: word, onto: base + suffix)
        }
        return word
    }

    private static func matchCase(of original: String, onto respelled: String) -> String {
        guard let first = original.first, first.isUppercase else { return respelled }
        return respelled.prefix(1).uppercased() + respelled.dropFirst()
    }

    /// Polish case/derivation endings these loanwords actually take.
    static let polishEndings: Set<String> = Set(rules.endings)

    /// The lexicon: common anglicisms → Polish phonetic respelling.
    ///
    /// Curated, not generated: every entry was chosen because the grapheme
    /// reading audibly fails and the respelling is the accepted spoken form.
    /// Words Poles already read correctly by Polish rules (laptop, internet,
    /// blog, film) are deliberately absent, respelling them would change
    /// nothing or make them worse.
    static let lexicon: [String: String] = rules.lexicon
}
