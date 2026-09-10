//! The one failure a caller has to tell from the others.
//!
//! The crate reports failures as `String`. Cancellation is not a fault: the
//! caller asked for it, and has to see that nothing was produced without
//! parsing a message. So it is one fixed string and one predicate.

/// The error [`Engine::synthesize`](crate::engine::Engine::synthesize) and
/// [`Engine::synthesize_window`](crate::engine::Engine::synthesize_window)
/// return when `Options::should_cancel` returned true. Nothing was produced.
pub const CANCELLED: &str = "cancelled: should_cancel returned true";

/// Whether `err` is the cancellation rather than a fault.
#[must_use]
pub fn is_cancelled(err: &str) -> bool {
    err == CANCELLED
}
