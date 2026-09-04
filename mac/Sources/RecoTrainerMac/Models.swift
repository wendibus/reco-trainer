import Foundation

enum Sport: String, Codable, CaseIterable, Identifiable, Sendable {
    case football
    case futsal
    case basketball
    case handball
    case hockey
    case rugby
    case lacrosse
    case americanFootball = "american_football"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .football:
            switch language {
            case .de: "Fußball"
            case .es: "Fútbol"
            case .en, .fr: "Football"
            }
        case .futsal: "Futsal"
        case .basketball: "Basketball"
        case .handball: "Handball"
        case .hockey: "Hockey"
        case .rugby: "Rugby"
        case .lacrosse: "Lacrosse"
        case .americanFootball:
            switch language {
            case .de, .en: "American Football"
            case .es: "Fútbol americano"
            case .fr: "Football américain"
            }
        }
    }

    var categories: [String] {
        switch self {
        case .football, .futsal: ["ball", "player", "goalkeeper", "referee", "goal"]
        case .basketball: ["ball", "player", "referee", "hoop"]
        case .handball: ["ball", "player", "goalkeeper", "referee", "goal"]
        case .hockey: ["puck", "player", "goalkeeper", "referee", "goal"]
        case .rugby, .americanFootball: ["ball", "player", "referee", "goalpost"]
        case .lacrosse: ["ball", "player", "goalkeeper", "referee", "goal"]
        }
    }
}

struct BoxAnnotation: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var category: String
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var confidence: Double?
    var source: String = "manual"
}

struct FrameRecord: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var relativePath: String
    var videoID: String
    var videoName: String
    var timestamp: Double
    var width: Int
    var height: Int
    var annotations: [BoxAnnotation] = []
    var reviewStatus: String?
}

struct ProjectDocument: Codable, Equatable, Sendable {
    static let schemaVersion = 2

    var schemaVersion = ProjectDocument.schemaVersion
    var name: String
    var sport: Sport
    var sourceFolder: String
    var createdAt = Date()
    var updatedAt = Date()
    var framesPerVideo: Int? = 240
    var frames: [FrameRecord] = []
    var lastTraining: TrainingResult?
    var trainingHistory: [TrainingResult]?
}

struct TrainingResult: Codable, Equatable, Sendable {
    var completedAt: String
    var model: String
    var device: String
    var epochsRequested: Int
    var frameCount: Int
    var annotatedFrames: Int
    var annotationCount: Int
    var classes: [String]
    var splits: [String: Int]
    var independentTest: Bool
    var checkpoint: String?
    var continuedFrom: String?
    var validationMetrics: [String: Double]?
    var testMetrics: [String: Double]?
}

struct VideoSource: Identifiable, Hashable, Sendable {
    let url: URL
    var id: String { url.path }
    var name: String { url.lastPathComponent }
}

enum ModelSize: String, CaseIterable, Identifiable, Sendable {
    case nano
    case small

    var id: String { rawValue }
    func title(language: AppLanguage) -> String { rawValue.capitalized }
}

struct HardwareStatus: Codable, Sendable {
    var python: String
    var platform: String
    var machine: String
    var torchInstalled: Bool
    var rfdetrInstalled: Bool
    var mpsAvailable: Bool
    var recommendedDevice: String
    var cpuCores: Int?
    var memoryGB: Int?
    var trainingBatchSize: Int?
    var dataWorkers: Int?
}

struct ActiveModelRecord: Codable, Sendable {
    var packageID: String
    var modelSize: String
    var sport: String
    var description: String?
    var weights: String
    var activatedAt: String
}

struct InstalledModelResponse: Codable, Sendable {
    var installed: Bool
    var active: ActiveModelRecord
}

struct ManagedModelRecord: Codable, Identifiable, Sendable {
    var packageID: String
    var displayName: String?
    var createdAt: String?
    var sport: String
    var modelSize: String
    var classes: [String]
    var source: String?
    var description: String?
    var validationMetrics: [String: Double]?
    var testMetrics: [String: Double]?
    var trainingSummary: ManagedTrainingSummary?
    var statistics: [String: Int]?
    var isActive: Bool?
    var isBest: Bool?

    var id: String { packageID }

    var validationScore: Double? {
        guard let metrics = validationMetrics ?? trainingSummary?.validationMetrics else { return nil }
        return Self.score(in: metrics)
    }

    var testScore: Double? {
        guard let metrics = testMetrics ?? trainingSummary?.testMetrics else { return nil }
        return Self.score(in: metrics)
    }

    var comparisonScore: Double? { testScore ?? validationScore }

    private static func score(in metrics: [String: Double]) -> Double? {
        return metrics.first { key, _ in
            let normalized = key.lowercased()
                .replacingOccurrences(of: "val/", with: "")
                .replacingOccurrences(of: "val_", with: "")
                .replacingOccurrences(of: "test/", with: "")
                .replacingOccurrences(of: "test_", with: "")
            return ["map_50_95", "map50_95", "ap/ball", "ap_ball"].contains(normalized)
        }?.value
    }
}

struct ManagedTrainingSummary: Codable, Sendable {
    var validationMetrics: [String: Double]?
    var testMetrics: [String: Double]?
}

struct ActivatedModelResponse: Codable, Sendable {
    var active: ActiveModelRecord
}

struct BenchmarkAnnotationRecord: Codable, Equatable, Sendable {
    var category: String
    var x: Double
    var y: Double
    var width: Double
    var height: Double
}

struct BenchmarkFrameRecord: Codable, Equatable, Sendable {
    var id: String
    var relativePath: String
    var width: Int
    var height: Int
    var annotations: [BenchmarkAnnotationRecord]
}

struct BenchmarkDatasetIdentity: Codable, Sendable {
    var sport: String
    var frames: [BenchmarkFrameRecord]
}

struct BenchmarkGroundTruthRecord: Codable, Sendable {
    var schemaVersion: Int
    var createdAt: String
    var sport: String
    var datasetID: String
    var frames: [BenchmarkFrameRecord]
    var reviewStatement: String

    var annotationCount: Int { frames.reduce(0) { $0 + $1.annotations.count } }
    var classes: [String] { Array(Set(frames.flatMap(\.annotations).map(\.category))).sorted() }
}

struct BenchmarkMetrics: Codable, Sendable {
    var qualityScore: Double
    var mAP50: Double
    var precision: Double
    var recall: Double
    var f1: Double
    var meanIoU: Double
    var truePositives: Int
    var falsePositives: Int
    var falseNegatives: Int
}

struct BenchmarkResult: Codable, Identifiable, Sendable {
    var rank: Int?
    var packageID: String
    var modelSize: String?
    var status: String
    var metrics: BenchmarkMetrics?
    var meanLatencyMs: Double?
    var error: String?
    var id: String { packageID }
}

struct BenchmarkReport: Codable, Sendable {
    var schemaVersion: Int
    var runID: String
    var createdAt: String
    var sport: String
    var datasetID: String
    var frameCount: Int
    var annotationCount: Int
    var modelCount: Int
    var successfulModelCount: Int
    var threshold: Double
    var device: String
    var rankingMethod: String
    var results: [BenchmarkResult]
}
