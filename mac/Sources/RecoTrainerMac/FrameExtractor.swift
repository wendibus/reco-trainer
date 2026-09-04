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

            for (index, seconds) in times.enumerated() {
                try Task.checkCancellation()
                let time = CMTime(seconds: seconds, preferredTimescale: 600)
                let image = try await generator.image(at: time).image
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
                let local = Double(index + 1) / Double(max(times.count, 1))
                await progress(min((completedDuration + duration * local) / totalDuration, 1), video.name)
            }
            completedDuration += duration
        }
        return result
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
