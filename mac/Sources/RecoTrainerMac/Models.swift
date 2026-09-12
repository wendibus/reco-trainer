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
    var fieldGeometry: FieldGeometry?

    var classes: [String] { Array(Set(frames.flatMap(\.annotations).map(\.category))).sorted() }
}

/// A single marked field corner, stored as a plain [x, y] pair (not CGPoint's
/// native {"x":...,"y":...} Codable form) so it matches what ml_worker.py's
/// field_membership_checker() expects to read directly with `for fx, fy in corners`.
struct FieldCorner: Codable, Equatable, Sendable {
    var x: Double
    var y: Double

    init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        x = try container.decode(Double.self)
        y = try container.decode(Double.self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(x)
        try container.encode(y)
    }
}

/// The project's marked field boundaries: four image corners in TL, TR, BR, BL
/// order, as fractional (0...1) coordinates of whichever reference frame they
/// were marked on (so they keep applying if frames are re-extracted at a
/// different resolution), plus the field's real width/length in meters.
/// auto_label() uses this to only consider people standing on the field.
struct FieldGeometry: Codable, Equatable, Sendable {
    var corners: [FieldCorner]
    var realWidth: Double
    var realLength: Double
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

struct PerClassBenchmarkMetrics: Codable, Sendable {
    var groundTruth: Int
    var predictions: Int
    var truePositives: Int
    var falsePositives: Int
    var falseNegatives: Int
    var precision: Double
    var recall: Double
    var f1: Double
    var ap50: Double
    var meanIoU: Double
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
    /// Absent from a benchmark report saved by an older Reco Trainer version - decoded
    /// as an empty dictionary rather than failing the whole report, so previously saved
    /// comparisons keep loading until the user reruns the benchmark.
    var perClass: [String: PerClassBenchmarkMetrics]

    enum CodingKeys: String, CodingKey {
        case qualityScore, mAP50, precision, recall, f1, meanIoU, truePositives, falsePositives, falseNegatives, perClass
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        qualityScore = try container.decode(Double.self, forKey: .qualityScore)
        mAP50 = try container.decode(Double.self, forKey: .mAP50)
        precision = try container.decode(Double.self, forKey: .precision)
        recall = try container.decode(Double.self, forKey: .recall)
        f1 = try container.decode(Double.self, forKey: .f1)
        meanIoU = try container.decode(Double.self, forKey: .meanIoU)
        truePositives = try container.decode(Int.self, forKey: .truePositives)
        falsePositives = try container.decode(Int.self, forKey: .falsePositives)
        falseNegatives = try container.decode(Int.self, forKey: .falseNegatives)
        perClass = try container.decodeIfPresent([String: PerClassBenchmarkMetrics].self, forKey: .perClass) ?? [:]
    }
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
    var classes: [String]
    var results: [BenchmarkResult]
}
