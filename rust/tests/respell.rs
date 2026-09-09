//! Polish lexical respelling: bit-parity checks against the Python/JS/Go
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

/// The digit branch is guarded on "no character is a letter", not on "every
/// character is a digit".
///
/// The word collector keeps `'` and `’` inside a word, because "deadline'u" is
/// one token to a Polish reader and its ending has to survive the respelling.
/// A quoted number therefore arrives as `'192`: letterless, and not all
/// digits. Under the digits guard it fell through every branch and came back
/// written, where the reference, JS and Swift all read it. A quoted IP address
/// or version number in Polish text is the shape that reaches it.
#[test]
fn a_quoted_number_is_read_the_way_a_bare_one_is() {
    assert_eq!(
        speech_text("Wpisz '192.168.0.1' w przeglądarce.", "pl"),
        "Wpisz ' jeden dziewięć dwa.sto sześćdziesiąt osiem przecinek zero.jeden ' w przeglądarce."
    );
    assert_eq!(
        speech_text("wersja '10.15.7' systemu", "pl"),
        "wersja ' jeden zero.piętnaście.siedem ' systemu"
    );
    // Digit by digit, because the quotes make the token something other than a
    // plain number: but never to *nothing*, so a character the table has no
    // word for stands for itself rather than being dropped.
    assert_eq!(lexical_respelling("'192'", "pl"), "' jeden dziewięć dwa '");
    assert_eq!(lexical_respelling("'07", "pl"), "' zero siedem");
    // The apostrophe the collector exists for still belongs to its word.
    assert_eq!(lexical_respelling("deadline'u", "pl"), "dedlajnu");
}

/// The respeller does not own this decision. It sees one word at a time, so it
/// cannot tell an initialism from a shout and would spell "TO JEST WAŻNE"
/// letter by letter. `spell_acronyms` decides for all twelve languages while
/// the surrounding capitals are still visible; the respeller sees the
/// already-spelled lowercase form and leaves it alone.
#[test]
fn acronyms_are_spelled_earlier_in_the_funnel() {
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
