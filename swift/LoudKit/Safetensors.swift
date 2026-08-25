import Foundation

/// Minimal safetensors reader: 8-byte little-endian header length, a JSON
/// header mapping tensor names to `{dtype, shape, data_offsets}`, then the
/// raw payload. The file is memory-mapped, so opening a 1.3 GB checkpoint
/// costs nothing until tensors are actually read — the same lazy contract as
/// `loudkit.checkpoint.Checkpoint`.
public final class Safetensors {
    public struct TensorInfo {
        public let dtype: String
        public let shape: [Int]
        let begin: Int
        let end: Int
    }

    private let data: Data
    private let payloadOffset: Int
    public private(set) var tensors: [String: TensorInfo] = [:]
    public private(set) var metadata: [String: String] = [:]

    public init(url: URL) throws {
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count > 8 else { throw LoudKitError.asset("\(url.lastPathComponent): truncated") }
        let headerLen = data.subdata(in: 0..<8).withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        }
        // Bounds-checked as UInt64 *before* the conversion. `Int(headerLen)`
        // traps for anything >= 2^63, so a corrupt or hostile file did not get
        // the refusal below — it killed the process with SIGTRAP, inside the
        // check meant to protect against it. This reader is reached by
        // `VoiceProfile.load`, i.e. by any downloaded `.voice.safetensors`.
        // Rust, Go and JS all survive the same header. The existing test used
        // `UInt64(1 << 40)`, which is under `Int.max`, so it passed.
        guard headerLen <= UInt64(data.count - 8) else {
            throw LoudKitError.asset("\(url.lastPathComponent): header overruns file")
        }
        payloadOffset = 8 + Int(headerLen)
        let headerData = data.subdata(in: 8..<payloadOffset)
        guard let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw LoudKitError.asset("\(url.lastPathComponent): header is not a JSON object")
        }
        for (name, value) in header {
            if name == "__metadata__" {
                metadata = value as? [String: String] ?? [:]
                continue
            }
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = (entry["shape"] as? [NSNumber])?.map({ $0.intValue }),
                  let offsets = (entry["data_offsets"] as? [NSNumber])?.map({ $0.intValue }),
                  offsets.count == 2 else {
                throw LoudKitError.asset("\(url.lastPathComponent): bad entry for \(name)")
            }
            // Validated here, where the other header checks live, because
            // `raw` slices `data[begin..<end]` and a Data subscript out of
            // range traps — the process dies rather than throwing. The offsets
            // come straight from the file, so a truncated or corrupt
            // checkpoint must be an error, not a crash.
            // Bounded *before* the addition, for the same reason `headerLen`
            // is bounded before its conversion a few lines up: Swift traps on
            // integer overflow, so `data_offsets: [9223372036854775807, …]`
            // killed the process inside the check written to refuse it. The
            // remaining budget is `data.count - payloadOffset`, and
            // `payloadOffset <= data.count` is already established, so this
            // subtraction cannot go negative and neither addition can overflow.
            guard offsets[0] >= 0, offsets[1] >= offsets[0],
                  offsets[1] <= data.count - payloadOffset else {
                throw LoudKitError.asset(
                    "\(url.lastPathComponent): \(name) has offsets outside the file")
            }
            let begin = offsets[0] + payloadOffset
            let end = offsets[1] + payloadOffset
            guard begin <= end, end <= data.count else {
                throw LoudKitError.asset(
                    "\(url.lastPathComponent): tensor \(name) spans \(offsets[0])..<\(offsets[1]) "
                        + "of a \(data.count - payloadOffset)-byte payload — file is truncated "
                        + "or the header is corrupt")
            }
            // The shape must account for exactly the bytes claimed.
            //
            // The range check above keeps `raw` from trapping, but callers
            // read `shape` to size their work: `VoiceProfile` checks a
            // dimension and `TokenGenerator` passes m/n/k straight to
            // `cblas_sgemm`, which reads what it is told to read. A header
            // declaring shape [256] over four bytes of payload is therefore
            // not a bad tensor, it is an out-of-bounds read in C with no
            // bounds check anywhere on the path. Overflow is checked rather
            // than assumed: `[1<<40, 1<<40]` multiplies to something small and
            // plausible if it is allowed to wrap.
            var elements = 1
            for dim in shape {
                guard dim >= 0 else {
                    throw LoudKitError.asset(
                        "\(url.lastPathComponent): tensor \(name) has a negative dimension in "
                            + "\(shape)")
                }
                let (product, overflowed) = elements.multipliedReportingOverflow(by: dim)
                guard !overflowed else {
                    throw LoudKitError.asset(
                        "\(url.lastPathComponent): tensor \(name) shape \(shape) overflows")
                }
                elements = product
            }
            guard let width = Self.byteWidth(of: dtype) else {
                throw LoudKitError.asset(
                    "\(url.lastPathComponent): tensor \(name) has unknown dtype \(dtype)")
            }
            let (declared, widthOverflowed) = elements.multipliedReportingOverflow(by: width)
            guard !widthOverflowed, declared == end - begin else {
                throw LoudKitError.asset(
                    "\(url.lastPathComponent): tensor \(name) declares shape \(shape) of "
                        + "\(dtype) (\(widthOverflowed ? "overflow" : String(declared)) bytes) but "
                        + "occupies \(end - begin) bytes — the header does not describe the payload")
            }
            tensors[name] = TensorInfo(dtype: dtype, shape: shape, begin: begin, end: end)
        }
    }

    /// Bytes per element, or nil for a dtype this reader does not know.
    ///
    /// Listed rather than inferred: an unknown dtype must be refused at load,
    /// not discovered later by a reader that happens to be asked for it.
    static func byteWidth(of dtype: String) -> Int? {
        switch dtype {
        case "F64", "I64", "U64": return 8
        case "F32", "I32", "U32": return 4
        case "F16", "BF16", "I16", "U16": return 2
        case "I8", "U8", "BOOL": return 1
        default: return nil
        }
    }

    public func names(prefix: String = "") -> [String] {
        tensors.keys.filter { $0.hasPrefix(prefix) }.sorted()
    }

    private func raw(_ name: String) throws -> (TensorInfo, Data) {
        guard let info = tensors[name] else {
            throw LoudKitError.asset("tensor not found: \(name)")
        }
        return (info, data.subdata(in: info.begin..<info.end))
    }

    /// Tensor as `[Float]`, upcasting F16 storage exactly (fp16 -> fp32 is a
    /// widening conversion, so the packed value survives untouched).
    public func floats(_ name: String) throws -> [Float] {
        let (info, bytes) = try raw(name)
        switch info.dtype {
        // `loadUnaligned`, not `bindMemory`. A tensor starts wherever the
        // header ends, so these slices carry no alignment guarantee — and
        // `bindMemory` requires one. It works on ARM, which tolerates unaligned
        // loads, and is undefined behaviour by Swift's own rules: the optimiser
        // is entitled to assume the alignment it was promised. A safetensors
        // file is data from outside the process; this is the one reader that
        // sees it first.
        case "F32":
            return bytes.withUnsafeBytes { buf in
                (0..<(buf.count / MemoryLayout<Float32>.size)).map {
                    buf.loadUnaligned(
                        fromByteOffset: $0 * MemoryLayout<Float32>.size, as: Float32.self)
                }
            }
        case "F16":
            return bytes.withUnsafeBytes { buf in
                (0..<(buf.count / MemoryLayout<Float16>.size)).map {
                    Float(
                        buf.loadUnaligned(
                            fromByteOffset: $0 * MemoryLayout<Float16>.size, as: Float16.self))
                }
            }
        default:
            throw LoudKitError.asset("\(name): dtype \(info.dtype) is not a float type this reader converts")
        }
    }

    public func ints(_ name: String) throws -> [Int] {
        let (info, bytes) = try raw(name)
        switch info.dtype {
        // `loadUnaligned` here for the same reason as `floats` above, and this
        // reader was left on `bindMemory` when that one was fixed: a tensor
        // starts wherever the header ends, so the slice carries no alignment
        // guarantee and `bindMemory` requires one. It happens to work on ARM
        // and is undefined behaviour by Swift's own rules — the optimiser is
        // entitled to assume the alignment it was promised, and a safetensors
        // file is data from outside the process.
        case "I64":
            return bytes.withUnsafeBytes { buf in
                (0..<(buf.count / MemoryLayout<Int64>.size)).map {
                    Int(
                        buf.loadUnaligned(
                            fromByteOffset: $0 * MemoryLayout<Int64>.size, as: Int64.self))
                }
            }
        case "I32":
            return bytes.withUnsafeBytes { buf in
                (0..<(buf.count / MemoryLayout<Int32>.size)).map {
                    Int(
                        buf.loadUnaligned(
                            fromByteOffset: $0 * MemoryLayout<Int32>.size, as: Int32.self))
                }
            }
        default:
            throw LoudKitError.asset("\(name): dtype \(info.dtype) is not an int type this reader converts")
        }
    }

    public func shape(_ name: String) throws -> [Int] {
        guard let info = tensors[name] else {
            throw LoudKitError.asset("tensor not found: \(name)")
        }
        return info.shape
    }
}
