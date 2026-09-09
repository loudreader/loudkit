import Foundation

/// Minimal safetensors reader: 8-byte little-endian header length, a JSON
/// header mapping tensor names to `{dtype, shape, data_offsets}`, then the
/// raw payload. The file is memory-mapped, so opening a 1.3 GB checkpoint
/// costs nothing until tensors are actually read, the same lazy contract as
/// `loudkit.checkpoint.Checkpoint`.
public final class Safetensors {
    /// Where one tensor is in the payload, and what shape it claims.
    public struct TensorInfo: Sendable, Equatable {
        /// The safetensors dtype string, `F32` and `I64` among them.
        public let dtype: String
        /// Dimensions, outermost first.
        public let shape: [Int]
        let begin: Int
        let end: Int
    }

    private let data: Data
    private let payloadOffset: Int
    /// Every tensor in the file, by name.
    public let tensors: [String: TensorInfo]
    /// The header's `__metadata__` block. The checkpoint manifest lives here.
    public let metadata: [String: String]

    /// Map a safetensors file and parse its header, reading no tensor.
    ///
    /// - Throws: `LoudKitError.asset` on a truncated file, a header that is not
    ///   JSON, or an offset that would read outside the payload.
    public init(url: URL) throws {
        data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count > 8 else { throw LoudKitError.asset("\(url.lastPathComponent): truncated") }
        let headerLen = data.subdata(in: 0..<8).withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self).littleEndian
        }
        // Bounds-checked as UInt64 *before* the conversion. `Int(headerLen)`
        // traps for anything >= 2^63, so a corrupt or hostile file did not get
        // the refusal below, it killed the process with SIGTRAP, inside the
        // check meant to protect against it. This reader is reached by
        // `VoiceProfile.load`, i.e. by any downloaded `.voice.safetensors`.
        // Rust, Go and JS all survive the same header. The existing test used
        // `UInt64(1 << 40)`, which is under `Int.max`, so it passed.
        guard headerLen <= UInt64(data.count - 8) else {
            throw LoudKitError.asset("\(url.lastPathComponent): header overruns file")
        }
        payloadOffset = 8 + Int(headerLen)
        let headerData = data.subdata(in: 8..<payloadOffset)
        // Last-wins on a repeated name, like the reference. See
        // `ManifestReader.json`: a repeated tensor entry here picks a
        // different slice of the payload in this port than in the other four.
        let headerRaw = try ManifestReader.json(
            headerData, "\(url.lastPathComponent): bad header JSON")
        guard let header = headerRaw as? [String: Any] else {
            throw LoudKitError.asset(
                "\(url.lastPathComponent): header is "
                + "\(ManifestReader.jsonTypePhrase(headerRaw)), expected a JSON object")
        }
        var readTensors: [String: TensorInfo] = [:]
        var readMetadata: [String: String] = [:]
        for (name, value) in header {
            if name == "__metadata__" {
                readMetadata = value as? [String: String] ?? [:]
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
            // range traps, the process dies rather than throwing. The offsets
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
                        + "of a \(data.count - payloadOffset)-byte payload: the file is truncated "
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
                        + "occupies \(end - begin) bytes: the header does not describe the payload")
            }
            readTensors[name] = TensorInfo(dtype: dtype, shape: shape, begin: begin, end: end)
        }
        tensors = readTensors
        metadata = readMetadata
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

    /// Tensor names beginning with `prefix`, sorted. An empty prefix is all
    /// of them.
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
        // header ends, so these slices carry no alignment guarantee, and
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

    /// One integer tensor, flattened, whatever width it was stored at.
    ///
    /// - Throws: `LoudKitError.asset` when the name is absent or the dtype is
    ///   not an integer one.
    public func ints(_ name: String) throws -> [Int] {
        let (info, bytes) = try raw(name)
        switch info.dtype {
        // `loadUnaligned` here for the same reason as `floats` above, and this
        // reader was left on `bindMemory` when that one was fixed: a tensor
        // starts wherever the header ends, so the slice carries no alignment
        // guarantee and `bindMemory` requires one. It happens to work on ARM
        // and is undefined behaviour by Swift's own rules, the optimiser is
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

    /// One tensor's declared shape, without reading its bytes.
    ///
    /// - Throws: `LoudKitError.asset` when the name is absent.
    public func shape(_ name: String) throws -> [Int] {
        guard let info = tensors[name] else {
            throw LoudKitError.asset("tensor not found: \(name)")
        }
        return info.shape
    }

    /// One tensor to write: its name, dtype, shape and little-endian bytes.
    public struct Entry: Sendable {
        /// The tensor's name in the file.
        public let name: String
        /// The safetensors dtype string to write.
        public let dtype: String
        /// Dimensions, outermost first. Their product must match `data`.
        public let shape: [Int]
        /// The payload, little-endian, already in the dtype's width.
        public let data: Data

        /// The four fields are all required: a writer that defaulted any of
        /// them could write a header that does not describe its own bytes.
        public init(name: String, dtype: String, shape: [Int], data: Data) {
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.data = data
        }
    }

    /// The order the safetensors library lays tensors out in: widest dtype
    /// first, then by name. Only the order the reader inverts.
    private static let dtypeRank = [
        "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16",
        "I32", "U32", "F32", "F64", "I64", "U64"
    ]

    /// Lay out tensors and metadata as a safetensors file: the 8-byte header
    /// length, the JSON header padded to a multiple of eight with spaces, then
    /// the payloads in header order. Owner-only permissions.
    public static func write(_ entries: [Entry], metadata: [String: String], to url: URL) throws {
        let sorted = entries.sorted { a, b in
            let (ra, rb) = (dtypeRank.firstIndex(of: a.dtype) ?? 0, dtypeRank.firstIndex(of: b.dtype) ?? 0)
            return ra != rb ? ra > rb : a.name < b.name
        }
        var fields: [String] = []
        if !metadata.isEmpty {
            let meta = try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys])
            guard let text = String(bytes: meta, encoding: .utf8) else {
                throw LoudKitError.asset("the safetensors metadata is not UTF-8")
            }
            fields.append("\"__metadata__\":" + text)
        }
        var offset = 0
        for e in sorted {
            guard let width = byteWidth(of: e.dtype) else {
                throw LoudKitError.asset("\(e.name): unknown dtype \(e.dtype)")
            }
            let elements = e.shape.reduce(1, *)
            guard elements * width == e.data.count else {
                throw LoudKitError.shape(
                    "\(e.name): shape \(e.shape) of \(e.dtype) is \(elements * width) bytes, "
                    + "data is \(e.data.count)")
            }
            guard let quoted = String(
                bytes: try JSONSerialization.data(withJSONObject: [e.name]), encoding: .utf8) else {
                throw LoudKitError.asset("\(e.name): the tensor name is not UTF-8")
            }
            let shape = e.shape.map(String.init).joined(separator: ",")
            fields.append(
                "\(quoted.dropFirst().dropLast()):{\"dtype\":\"\(e.dtype)\",\"shape\":[\(shape)],"
                + "\"data_offsets\":[\(offset),\(offset + e.data.count)]}")
            offset += e.data.count
        }
        var header = "{" + fields.joined(separator: ",") + "}"
        while header.utf8.count % 8 != 0 { header += " " }
        var out = Data()
        var length = UInt64(header.utf8.count).littleEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(contentsOf: Array(header.utf8))
        for e in sorted { out.append(e.data) }
        try out.write(to: url)
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
}
