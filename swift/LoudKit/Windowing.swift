import Foundation

/// The render window, and what happens when a passage does not fit it.
///
/// One definition, shared by every caller — `Engine.stripSpecials` and
/// `MelDecoder.decode` included. A passage longer than the window is refused,
/// not sliced: `.prefix(maxSpeechTokens)` leaves the end of a passage
/// nonexistent while the audio still sounds perfectly fine, which is silent
/// data loss — noticed only by a listener who knows the text. Python
/// (`engine.py:466`), Rust (`windowing.rs:97`), Go (`windowing.go:79`) and JS
/// (`windowing.ts:71`) all refuse it the same way.
///
/// It matters more here than anywhere else: this module has no
/// `synthesizeLong`, so without the refusal a caller handing it a paragraph
/// gets clipped audio and no error from any layer.
enum Windowing {
    /// Throw unless `count` speech tokens fit `maxSpeechTokens`.
    ///
    /// The message is the other ports\' message, word for word, so a user who
    /// hits it in two languages reads the same sentence twice.
    static func requireFits(_ count: Int, _ maxSpeechTokens: Int) throws {
        guard count > maxSpeechTokens else { return }
        throw LoudKitError.shape(
            "\(count) speech tokens exceed the \(maxSpeechTokens)-token window by "
                + "\(count - maxSpeechTokens); split the text first")
    }
}
