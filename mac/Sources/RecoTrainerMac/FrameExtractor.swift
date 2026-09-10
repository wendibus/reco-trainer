@preconcurrency import AVFoundation
import AppKit
import CryptoKit
import Foundation

enum FrameExtractorError: LocalizedError {
    case noVideos
    case unreadableVideo(String)

    var errorDescription: String? {
        switch self {
        case .noVideos: "Im gewählten Ordner wurden keine unterstützten Videos gefunden."
        case .unreadableVideo(let name): "Das Video \(name) konnte nicht gelesen werden."
        }
    }
}

/// Thread-safe "how many of the N expected callbacks have fired" counter for
/// generateCGImagesAsynchronously's completion handler, which can be invoked from a
/// background queue outside Swift's structured concurrency.
private final class FrameCompletionCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var remaining: Int

    init(total: Int) { remaining = total }

    /// Returns true exactly once, when the last expected callback has fired.
    func decrementAndCheckFinished() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        remaining -= 1
        return remaining == 0
    }
}

struct FrameExtractor {
    static let supportedExtensions: Set<String> = ["mp4", "mov", "m4v"]

    func discoverVideos(in folder: URL) -> [VideoSource] {
        guard let enumerator = FileManager.default.enumerator(
            at: folder,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return [] }

        var videos: [VideoSource] = []
        while let url = enumerator.nextObject() as? URL {
            if url.lastPathComponent == ProjectStore.projectDirectoryName ||
                url.lastPathComponent == ProjectStore.visibleProjectDirectoryName {
                enumerator.skipDescendants()
                continue
            }
            guard Self.supportedExtensions.contains(url.pathExtension.lowercased()) else { continue }
            videos.append(VideoSource(url: url))
        }
        return videos.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    func extract(
        videos: [VideoSource],
        into store: ProjectStore,
        framesPerVideo: Int = 240,
        progress: @escaping @Sendable (Double, String) async -> Void
    ) async throws -> [FrameRecord] {
        guard !videos.isEmpty else { throw FrameExtractorError.noVideos }
        try store.prepare()

        var durations: [(VideoSource, Double)] = []
        for video in videos {
            let asset = AVURLAsset(url: video.url)
            let duration = try await asset.load(.duration).seconds
            guard duration.isFinite, duration > 0 else {
                throw FrameExtractorError.unreadableVideo(video.name)
            }
            durations.append((video, duration))
        }

        let totalDuration = max(durations.reduce(0) { $0 + $1.1 }, 1)
        var result: [FrameRecord] = []
        var completedDuration = 0.0

        for (video, duration) in durations {
            let targetCount = min(5_000, max(4, framesPerVideo))
            let spacing = max(0.5, duration / Double(targetCount))
            let times = stride(from: 0.25, to: max(duration - 0.1, 0.3), by: spacing).map { $0 }

            let asset = AVURLAsset(url: video.url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = CMTime(seconds: 0.12, preferredTimescale: 600)
            generator.requestedTimeToleranceAfter = CMTime(seconds: 0.12, preferredTimescale: 600)
            let videoID = stableID(for: video.url.path)

            var completedFrames = 0
            for try await (index, image) in generatedImages(for: generator, at: times, videoName: video.name) {
                try Task.checkCancellation()
                let seconds = times[index]
                let fileName = "\(videoID)-\(String(format: "%06d", index)).jpg"
                let outputURL = store.framesURL.appending(path: fileName)
                try writeJPEG(image, to: outputURL)

                result.append(FrameRecord(
                    relativePath: "frames/\(fileName)",
                    videoID: videoID,
                    videoName: video.name,
                    timestamp: seconds,
                    width: image.width,
                    height: image.height
                ))
                completedFrames += 1
                let local = Double(completedFrames) / Double(max(times.count, 1))
                await progress(min((completedDuration + duration * local) / totalDuration, 1), video.name)
            }
            completedDuration += duration
        }
        return result
    }

    /// Generates every requested frame with one batched, sequential decode pass instead
    /// of calling `image(at:)` once per frame. AVAssetImageGenerator re-seeks to the
    /// nearest keyframe and decodes forward for every individual `image(at:)` call, which
    /// made extracting e.g. 240 frames 240 separate seek-and-decode operations per video.
    /// `generateCGImagesAsynchronously(forTimes:)` lets it decode through the file once and
    /// hand back frames as their timestamps are reached, matching how the cross-platform
    /// ffmpeg-based extractor already does this in a single pass.
    private func generatedImages(
        for generator: AVAssetImageGenerator,
        at times: [Double],
        videoName: String
    ) -> AsyncThrowingStream<(index: Int, image: CGImage), Error> {
        let requestedTimes = times.map { CMTime(seconds: $0, preferredTimescale: 600) }
        let counter = FrameCompletionCounter(total: requestedTimes.count)
        return AsyncThrowingStream { continuation in
            generator.generateCGImagesAsynchronously(forTimes: requestedTimes.map(NSValue.init(time:))) { requestedTime, image, _, result, error in
                switch result {
                case .succeeded:
                    if let image, let index = requestedTimes.firstIndex(of: requestedTime) {
                        continuation.yield((index, image))
                    }
                case .failed:
                    continuation.finish(throwing: error ?? FrameExtractorError.unreadableVideo(videoName))
                    return
                case .cancelled:
                    break
                @unknown default:
                    break
                }
                if counter.decrementAndCheckFinished() {
                    continuation.finish()
                }
            }
            continuation.onTermination = { _ in generator.cancelAllCGImageGeneration() }
        }
    }

    private func stableID(for value: String) -> String {
        let digest = SHA256.hash(data: Data(value.utf8))
        return digest.prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    private func writeJPEG(_ image: CGImage, to url: URL) throws {
        let representation = NSBitmapImageRep(cgImage: image)
        guard let data = representation.representation(using: .jpeg, properties: [.compressionFactor: 0.88]) else {
            throw CocoaError(.fileWriteUnknown)
        }
        try data.write(to: url, options: .atomic)
    }
}
