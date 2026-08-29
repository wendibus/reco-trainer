import AppKit
import AVFoundation
import Foundation

if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--probe" {
    let input = URL(fileURLWithPath: CommandLine.arguments[2])
    let asset = AVURLAsset(url: input)
    let duration = CMTimeGetSeconds(asset.duration)
    guard let track = asset.tracks(withMediaType: .video).first, duration.isFinite, duration > 0 else { exit(3) }
    let transformed = track.naturalSize.applying(track.preferredTransform)
    let width = Int(abs(transformed.width))
    let height = Int(abs(transformed.height))
    print("{\"duration\":\(duration),\"width\":\(width),\"height\":\(height)}")
    exit(0)
}

guard CommandLine.arguments.count == 5 else {
    FileHandle.standardError.write(Data("Usage: frame_extractor INPUT OUTPUT_DIR PREFIX COUNT\n".utf8))
    exit(2)
}

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let outputDirectory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
let prefix = CommandLine.arguments[3]
guard let count = Int(CommandLine.arguments[4]), count > 0 else { exit(2) }

let asset = AVURLAsset(url: input)
let duration = CMTimeGetSeconds(asset.duration)
guard duration.isFinite, duration > 0 else {
    FileHandle.standardError.write(Data("Unreadable video duration\n".utf8))
    exit(3)
}

let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.requestedTimeToleranceBefore = CMTime(seconds: 0.12, preferredTimescale: 600)
generator.requestedTimeToleranceAfter = CMTime(seconds: 0.12, preferredTimescale: 600)
generator.maximumSize = CGSize(width: 1280, height: 1280)

try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

for index in 0..<count {
    autoreleasepool {
        let seconds = min(duration * (Double(index) + 0.5) / Double(count), max(duration - 0.01, 0))
        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        do {
            let image = try generator.copyCGImage(at: time, actualTime: nil)
            let bitmap = NSBitmapImageRep(cgImage: image)
            guard let data = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.88]) else {
                throw CocoaError(.fileWriteUnknown)
            }
            let name = String(format: "%@-%06d.jpg", prefix, index + 1)
            try data.write(to: outputDirectory.appendingPathComponent(name), options: .atomic)
        } catch {
            FileHandle.standardError.write(Data("Frame \(index + 1): \(error.localizedDescription)\n".utf8))
        }
    }
}
