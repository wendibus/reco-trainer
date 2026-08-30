import Foundation

enum Sport: String, Codable, CaseIterable, Identifiable {
    case football
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
        case .football: ["ball", "player", "goalkeeper", "referee", "goal"]
        case .basketball: ["ball", "player", "referee", "hoop"]
        case .handball: ["ball", "player", "goalkeeper", "referee", "goal"]
        case .hockey: ["puck", "player", "goalkeeper", "referee", "goal"]
        case .rugby, .americanFootball: ["ball", "player", "referee", "goalpost"]
        case .lacrosse: ["ball", "player", "goalkeeper", "referee", "goal"]
        }
    }
}

struct BoxAnnotation: Codable, Identifiable, Equatable {
    var id = UUID()
    var category: String
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var confidence: Double?
    var source: String = "manual"
}

struct FrameRecord: Codable, Identifiable, Equatable {
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

struct ProjectDocument: Codable, Equatable {
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

struct TrainingResult: Codable, Equatable {
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
    var validationMetrics: [String: Double]?
}

struct VideoSource: Identifiable, Hashable {
    let url: URL
    var id: String { url.path }
    var name: String { url.lastPathComponent }
}

enum ModelSize: String, CaseIterable, Identifiable {
    case nano
    case small

    var id: String { rawValue }
    func title(language: AppLanguage) -> String { rawValue.capitalized }
}

struct HardwareStatus: Codable {
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

struct ActiveModelRecord: Codable {
    var packageID: String
    var modelSize: String
    var sport: String
    var description: String?
    var weights: String
    var activatedAt: String
}

struct InstalledModelResponse: Codable {
    var installed: Bool
    var active: ActiveModelRecord
}

struct BenchmarkAnnotationRecord: Codable, Equatable {
    var category: String
    var x: Double
    var y: Double
    var width: Double
    var height: Double
}

struct BenchmarkFrameRecord: Codable, Equatable {
    var id: String
    var relativePath: String
    var width: Int
    var height: Int
    var annotations: [BenchmarkAnnotationRecord]
}

struct BenchmarkDatasetIdentity: Codable {
    var sport: String
    var frames: [BenchmarkFrameRecord]
}

struct BenchmarkGroundTruthRecord: Codable {
    var schemaVersion: Int
    var createdAt: String
    var sport: String
    var datasetID: String
    var frames: [BenchmarkFrameRecord]
    var reviewStatement: String

    var annotationCount: Int { frames.reduce(0) { $0 + $1.annotations.count } }
    var classes: [String] { Array(Set(frames.flatMap(\.annotations).map(\.category))).sorted() }
}

struct BenchmarkMetrics: Codable {
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

struct BenchmarkResult: Codable, Identifiable {
    var rank: Int?
    var packageID: String
    var modelSize: String?
    var status: String
    var metrics: BenchmarkMetrics?
    var meanLatencyMs: Double?
    var error: String?
    var id: String { packageID }
}

struct BenchmarkReport: Codable {
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
