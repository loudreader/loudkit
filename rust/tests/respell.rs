//! Polish lexical respelling — bit-parity checks against the Python/JS/Go
//! ports. The expected values are the ones the Swift/Python ear tests approved.

use loudkit::letters::spell_acronyms;
use loudkit::respell::lexical_respelling;
use loudkit::speechtext::speech_text;

#[test]
fn curated_lexicon() {
    for (word, want) in [
        ("download", "dałnloud"),
        ("deadline", "dedlajn"),
        ("feedback", "fidbek"),
        ("weekend", "łikend"),
        ("workflow", "łorkfloł"),
        ("release", "rilis"),
    ] {
        assert_eq!(lexical_respelling(word, "pl"), want, "{word}");
    }
}

#[test]
fn case_is_preserved() {
    assert_eq!(lexical_respelling("GitHub", "pl"), "Githab");
    assert_eq!(lexical_respelling("Download", "pl"), "Dałnloud");
}

#[test]
fn phrases_respell_as_a_unit() {
    assert_eq!(lexical_respelling("release notes", "pl"), "rilis nołc");
    assert_eq!(lexical_respelling("pull request", "pl"), "pul rekłest");
    assert_eq!(lexical_respelling("code review", "pl"), "koud riwju");
}

#[test]
fn only_polish_is_respelled() {
    assert_eq!(lexical_respelling("download", "en"), "download");
}

#[test]
fn numbers_become_cardinals() {
    assert_eq!(lexical_respelling("0", "pl"), "zero");
    assert_eq!(lexical_respelling("15", "pl"), "piętnaście");
    assert_eq!(lexical_respelling("101", "pl"), "sto jeden");
    assert_eq!(
        lexical_respelling("1234", "pl"),
        "tysiąc dwieście trzydzieści cztery"
    );
}

#[test]
fn decimals_read_whole_comma_fraction() {
    assert_eq!(lexical_respelling("2.5", "pl"), "dwa przecinek pięć");
}

/// The respeller no longer owns this decision. It saw one word at a time, so it
/// could not tell an initialism from a shout and spelled "TO JEST WAŻNE" letter
/// by letter. `spell_acronyms` decides for all twelve languages while the
/// surrounding capitals are still visible; the respeller now sees the
/// already-spelled lowercase form and leaves it alone.
#[test]
fn acronyms_are_spelled_earlier_in_the_funnel_now() {
    assert_eq!(spell_acronyms("GPT", "pl"), "gie-pe-te");
    assert_eq!(spell_acronyms("USB", "pl"), "u-es-be");
    // word-acronyms keep their word form
    assert_eq!(spell_acronyms("NASA", "pl"), "nasa");
    assert_eq!(spell_acronyms("PIN", "pl"), "pin");
    // and the whole funnel still produces the Polish letter names
    assert!(speech_text("Model GPT jest dobry.", "pl").contains("gie-pe-te"));
    // a run of capitals is emphasis, and the respeller must not undo that
    assert_eq!(speech_text("CIA CIA", "pl"), "CIA CIA");
}

#[test]
fn english_word_alone_stays_polish_in_span_transliterates() {
    assert_eq!(lexical_respelling("brown", "pl"), "brown");
    assert_eq!(
        lexical_respelling("the quick brown fox", "pl"),
        "da kłyk brałn faks"
    );
}

#[test]
fn inflection_via_stem() {
    assert_eq!(lexical_respelling("update", "pl"), "apdejt");
    assert_eq!(lexical_respelling("updates", "pl"), "apdejc");
    assert_eq!(lexical_respelling("deadline'u", "pl"), "dedlajnu");
}

#[test]
fn polish_words_are_untouched() {
    assert_eq!(lexical_respelling("temperatura", "pl"), "temperatura");
    assert_eq!(lexical_respelling("piątku", "pl"), "piątku");
}

#[test]
fn full_sentence_matches_other_ports() {
    assert_eq!(
        speech_text("Pobierz download i zrób code review.", "pl"),
        "Pobierz dałnloud i zrób koud riwju."
    );
    assert_eq!(
        speech_text("Rabat 15% na weekend!", "pl"),
        "Rabat piętnaście procent na łikend!"
    );
    assert_eq!(
        speech_text("The quick brown fox jumps over the lazy dog.", "pl"),
        "Da kłyk brałn faks dżamps ołwer da lejzi dog."
    );
    assert_eq!(
        speech_text("Skończ deadline'u przed piątkiem.", "pl"),
        "Skończ dedlajnu przed piątkiem."
    );
    assert_eq!(
        speech_text("GPT działa dobrze na USB.", "pl"),
        "gie-pe-te działa dobrze na u-es-be."
    );
    assert_eq!(
        speech_text("2.5 GB to dużo.", "pl"),
        "dwa przecinek pięć gie-be to dużo."
    );
}

#[test]
fn long_tail_is_loaded() {
    // entries from the generated 110k CMUdict lexicon
    assert_eq!(lexical_respelling("queue", "pl"), "kju");
    assert_eq!(lexical_respelling("thought", "pl"), "tot");
    assert_eq!(lexical_respelling("juice", "pl"), "dżus");
}
