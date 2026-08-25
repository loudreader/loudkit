import Foundation
import OSLog

/// The one logging call this target makes.
///
/// Kept local rather than exported: a logging façade on the text target
/// would make every consumer of the funnel depend on this package's choice of
/// logger, which is not a choice a text scrubber gets to make for its host.
enum Log {
    private static let logger = Logger(subsystem: "dev.loudkit.text", category: "respell")

    static func error(_ message: String, category: String = "respell") {
        logger.error("[\(category, privacy: .public)] \(message, privacy: .public)")
    }
}
