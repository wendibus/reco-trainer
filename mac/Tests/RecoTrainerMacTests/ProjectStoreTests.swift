import CoreGraphics
import Foundation
import Testing
@testable import RecoTrainerMac

@Test func projectRoundTrip() throws {
    let root = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let store = ProjectStore(rootURL: root)
    let frame = FrameRecord(
        relativePath: "frames/example.jpg",
        videoID: "video-a",
        videoName: "match.mov",
        timestamp: 12.5,
        width: 1920,
        height: 1080,
        annotations: [BoxAnnotation(category: "ball", x: 10, y: 20, width: 8, height: 8)]
    )
    let project = ProjectDocument(name: "Test", sport: .football, sourceFolder: "/tmp/videos", frames: [frame])

    try store.save(project)
    let loaded = try store.load()

    #expect(loaded.name == "Test")
    #expect(loaded.sport == .football)
    #expect(loaded.frames.count == 1)
    #expect(loaded.frames[0].annotations[0].category == "ball")
}

@Test func sportSchemasAreSpecialized() {
    #expect(Sport.basketball.categories.contains("hoop"))
    #expect(Sport.hockey.categories.first == "puck")
    #expect(!Sport.football.categories.contains("hoop"))
    #expect(Sport.rugby.categories.contains("goalpost"))
    #expect(Sport.lacrosse.categories.contains("goalkeeper"))
    #expect(Sport.americanFootball.rawValue == "american_football")
    #expect(Sport.americanFootball.categories.contains("goalpost"))
}

@Test func savingCreatesLocalProjectBackup() throws {
    let root = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-backup-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let store = ProjectStore(rootURL: root)
    var project = ProjectDocument(name: "Backup", sport: .basketball, sourceFolder: "/tmp/videos")
    try store.save(project)
    project.name = "Backup updated"
    try store.save(project)
    let backups = try FileManager.default.contentsOfDirectory(at: store.backupsURL, includingPropertiesForKeys: nil)
    #expect(backups.count == 1)
}

@Test func removingTrainingFrameKeepsSourceVideoAndInvalidatesBenchmark() throws {
    let folder = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-remove-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let store = ProjectStore.forSourceFolder(folder)
    try store.prepare()
    let frame = FrameRecord(relativePath: "frames/synthetic.jpg", videoID: "video", videoName: "source.mov", timestamp: 1, width: 10, height: 10)
    let sourceVideo = folder.appending(path: "source.mov")
    try Data("synthetic source".utf8).write(to: sourceVideo)
    try Data("synthetic frame".utf8).write(to: store.frameURL(for: frame))
    let benchmark = store.rootURL.appending(path: "benchmarks", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: benchmark, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: benchmark.appending(path: "ground-truth.json"))
    try Data("{}".utf8).write(to: benchmark.appending(path: "latest.json"))

    let invalidated = try store.removeDerivedFrame(frame)

    #expect(invalidated)
    #expect(FileManager.default.fileExists(atPath: sourceVideo.path))
    #expect(!FileManager.default.fileExists(atPath: store.frameURL(for: frame).path))
    #expect(!FileManager.default.fileExists(atPath: benchmark.appending(path: "ground-truth.json").path))
}

@Test func videoDiscoverySkipsRecoTrainingDirectories() throws {
    let folder = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-discovery-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let store = ProjectStore.forSourceFolder(folder)
    try store.prepare()

    let sourceVideo = folder.appending(path: "match.mov")
    let derivedVideo = store.framesURL.appending(path: "must-not-be-discovered.mov")
    try Data().write(to: sourceVideo)
    try Data().write(to: derivedVideo)

    let videos = FrameExtractor().discoverVideos(in: folder)

    #expect(videos.map(\.url.standardizedFileURL) == [sourceVideo.standardizedFileURL])
}

@Test @MainActor func choosingVideoFolderUsesLocalPicker() {
    let app = AppState()

    app.chooseFolder()

    #expect(app.localPickerPurpose == .trainingFolder)
}

// autoLabelCategories drives the multi-select "Automatisch markieren" chips and is
// deliberately separate from selectedCategory (used for hand-drawing a box and the
// active-learning Ball/No-ball review, both inherently single-category).
@Test @MainActor func autoLabelCategoriesDefaultsToBallAndSupportsMultiSelection() {
    let app = AppState()

    #expect(app.autoLabelCategories == ["ball"])

    app.autoLabelCategories.insert("player")
    #expect(app.autoLabelCategories == ["ball", "player"])

    app.autoLabelCategories.remove("ball")
    #expect(app.autoLabelCategories == ["player"])

    // selectedCategory (single-select, used for drawing) is unaffected by autoLabelCategories.
    #expect(app.selectedCategory == "ball")
}

// Regression coverage for a real point of confusion: selecting "hoop" in the
// auto-label chips silently did nothing (base model only knows COCO's "sports
// ball" and "person"), with the explanation buried in the log instead of visible
// at the chip. autoLabelSupportedCategories is what the chip UI now uses to grey
// those out up front. "referee" is included even without a custom model: the
// clothing heuristic (ml_worker.py's REFEREE_CLOTHING_PROFILES) can reclassify a
// generic "player" detection to "referee" for every sport this app supports.
@Test @MainActor func autoLabelSupportedCategoriesExcludesUnknownClassesWithoutACustomModel() {
    let app = AppState()
    app.sport = .basketball

    #expect(app.autoLabelSupportedCategories == ["ball", "player", "referee"])
    #expect(!app.autoLabelSupportedCategories.contains("hoop"))
}

// Regression coverage for a real report: a model activated back when the project
// only had "ball" annotated used to have "every sport category" assumed supported
// just because SOME custom checkpoint existed, so auto-label wastefully ran
// inference for "referee"/"player" too and reported a single misleading "0 boxes"
// instead of explaining that this specific model never learned those classes.
// autoLabelSupportedCategories must defer to the activated model's own class list -
// except "player" (base-model "person" fallback) and "referee" (clothing
// heuristic), which stay available regardless of what the active model itself
// was trained on, since neither needs custom training to be detected at all.
@Test @MainActor func autoLabelSupportedCategoriesUsesTheActiveModelsOwnClassListWhenOneExists() {
    let app = AppState()
    app.sport = .basketball
    app.modelSize = .nano
    app.activeModelPackageID = "pkg-1"
    app.managedModels = [
        ManagedModelRecord(
            packageID: "pkg-1", displayName: nil, createdAt: nil, sport: "basketball",
            modelSize: "nano", classes: ["ball"], source: nil, description: nil,
            validationMetrics: nil, testMetrics: nil, trainingSummary: nil,
            statistics: nil, isActive: true, isBest: nil
        )
    ]

    #expect(app.autoLabelSupportedCategories == ["ball", "player", "referee"])
}

// Regression coverage: a freshly loaded project used to preselect only the sport's
// first category (typically "ball"), so getting person detections for free from the
// base model's generic COCO "person" class required remembering to also tick "player" -
// referees then had to be drawn by hand from scratch instead of being generically
// detected as "player" and just relabeled via the annotation editor. Loading a project
// now preselects every base-model (or custom-model) supported category instead.
@Test func defaultAutoLabelCategoriesPreselectsEveryCurrentlySupportedCategory() {
    #expect(AppState.defaultAutoLabelCategories(selectedCategory: "ball", supported: ["ball", "player"]) == ["ball", "player"])
    // Falls back to just the selected category if nothing is determinable as supported.
    #expect(AppState.defaultAutoLabelCategories(selectedCategory: "ball", supported: []) == ["ball"])
}

// annotation(at:among:) backs click-to-select in the annotation editor. Preferring the
// smallest containing box keeps a small box nested inside a larger one - e.g. a ball
// box inside a player box - individually selectable instead of always hitting the box
// underneath it.
@Test func annotationHitTestPrefersTheSmallestContainingBox() {
    let bigBox = BoxAnnotation(category: "player", x: 0, y: 0, width: 100, height: 100)
    let smallBox = BoxAnnotation(category: "ball", x: 40, y: 40, width: 10, height: 10)
    let annotations = [bigBox, smallBox]

    #expect(annotation(at: CGPoint(x: 45, y: 45), among: annotations)?.id == smallBox.id)
    #expect(annotation(at: CGPoint(x: 5, y: 5), among: annotations)?.id == bigBox.id)
    #expect(annotation(at: CGPoint(x: 200, y: 200), among: annotations) == nil)
}

// clampedMove(...) backs drag-to-move: a box dragged past an edge should stop at the
// edge instead of moving partly or fully outside the frame.
@Test func clampedMoveKeepsTheBoxInsideTheFrame() {
    let box = BoxAnnotation(category: "player", x: 10, y: 10, width: 20, height: 20)

    let pastTopLeft = clampedMove(of: box, byImageDelta: CGSize(width: -50, height: -50), imageWidth: 100, imageHeight: 100)
    #expect(pastTopLeft.x == 0)
    #expect(pastTopLeft.y == 0)

    let pastBottomRight = clampedMove(of: box, byImageDelta: CGSize(width: 200, height: 200), imageWidth: 100, imageHeight: 100)
    #expect(pastBottomRight.x == 80)
    #expect(pastBottomRight.y == 80)

    let withinBounds = clampedMove(of: box, byImageDelta: CGSize(width: 5, height: 5), imageWidth: 100, imageHeight: 100)
    #expect(withinBounds.x == 15)
    #expect(withinBounds.y == 15)
}

// Regression coverage: the "Freeze reviewed answers" button used to stay disabled
// whenever ANY frame in the whole project had an unreviewed automatic annotation,
// including pending active-learning review candidates. Those candidate frames are
// excluded from ground truth by freezeBenchmarkGroundTruth() itself, so they should
// never be able to block freezing - only unreviewed training frames should.
@Test @MainActor func benchmarkFreezeIgnoresUnreviewedActiveLearningCandidates() {
    let app = AppState()
    let reviewedTrainingFrame = FrameRecord(
        relativePath: "frames/reviewed.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 0,
        width: 10, height: 10,
        annotations: [BoxAnnotation(category: "ball", x: 1, y: 1, width: 2, height: 2, source: "manual")]
    )
    var pendingCandidateFrame = FrameRecord(
        relativePath: "frames/candidate.jpg", videoID: "v2", videoName: "clip2.mov", timestamp: 0,
        width: 10, height: 10,
        annotations: [BoxAnnotation(category: "ball", x: 1, y: 1, width: 2, height: 2, source: "auto")]
    )
    pendingCandidateFrame.reviewStatus = "candidate"
    app.project = ProjectDocument(name: "Test", sport: .basketball, sourceFolder: "/tmp/videos", frames: [reviewedTrainingFrame, pendingCandidateFrame])

    #expect(app.hasUnreviewedTrainingAnnotations == false)
}

// Regression coverage: the annotation editor has no way to select, move, or resize
// an individual box, so bulk accept/reject is the only mechanism that can ever clear
// source == "auto" on a regular training frame (as opposed to an active-learning
// candidate, which has its own separate Ball/No-ball review flow). Without this,
// hasUnreviewedTrainingAnnotations can never become false once autolabel has run,
// permanently blocking step 1 of the benchmark workflow.
@Test @MainActor func acceptingAutomaticAnnotationsMarksThemManual() throws {
    let folder = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-accept-auto-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let app = AppState()
    app.selectedFolder = folder
    let frame = FrameRecord(
        relativePath: "frames/one.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 0,
        width: 10, height: 10,
        annotations: [
            BoxAnnotation(category: "ball", x: 1, y: 1, width: 2, height: 2, source: "auto"),
            BoxAnnotation(category: "ball", x: 4, y: 4, width: 2, height: 2, source: "manual"),
        ]
    )
    app.project = ProjectDocument(name: "Test", sport: .basketball, sourceFolder: folder.path, frames: [frame])
    app.selectedFrameID = frame.id

    #expect(app.automaticAnnotationCountOnSelectedFrame == 1)

    app.acceptAllAutomaticAnnotations()

    #expect(app.automaticAnnotationCountOnSelectedFrame == 0)
    #expect(app.project?.frames.first?.annotations.count == 2)
    #expect(app.project?.frames.first?.annotations.allSatisfy { $0.source == "manual" } == true)
}

@Test @MainActor func rejectingAutomaticAnnotationsRemovesOnlyAutoBoxes() throws {
    let folder = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-reject-auto-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let app = AppState()
    app.selectedFolder = folder
    let frame = FrameRecord(
        relativePath: "frames/one.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 0,
        width: 10, height: 10,
        annotations: [
            BoxAnnotation(category: "ball", x: 1, y: 1, width: 2, height: 2, source: "auto"),
            BoxAnnotation(category: "ball", x: 4, y: 4, width: 2, height: 2, source: "manual"),
        ]
    )
    app.project = ProjectDocument(name: "Test", sport: .basketball, sourceFolder: folder.path, frames: [frame])
    app.selectedFrameID = frame.id

    app.rejectAllAutomaticAnnotations()

    #expect(app.project?.frames.first?.annotations.count == 1)
    #expect(app.project?.frames.first?.annotations.first?.source == "manual")
}

@Test @MainActor func benchmarkFreezeBlocksOnUnreviewedTrainingAnnotations() {
    let app = AppState()
    let unreviewedTrainingFrame = FrameRecord(
        relativePath: "frames/unreviewed.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 0,
        width: 10, height: 10,
        annotations: [BoxAnnotation(category: "ball", x: 1, y: 1, width: 2, height: 2, source: "auto")]
    )
    app.project = ProjectDocument(name: "Test", sport: .basketball, sourceFolder: "/tmp/videos", frames: [unreviewedTrainingFrame])

    #expect(app.hasUnreviewedTrainingAnnotations == true)
}

// Regression coverage: BenchmarkMetrics.perClass is what backs the per-category
// comparison table (mAP@.50 broken down by class instead of only the overall
// aggregate, so it's visible when one model is stronger at e.g. ball and another
// at referee). A report saved by an older Reco Trainer version has no "perClass"
// key at all - it must still decode instead of losing the whole benchmark report.
@Test func benchmarkMetricsDecodesPerClassAndFallsBackWhenAbsent() throws {
    let decoder = JSONDecoder()

    let withPerClass = """
    {
        "qualityScore": 88.5, "mAP50": 0.9, "precision": 0.95, "recall": 0.92, "f1": 0.93,
        "meanIoU": 0.8, "truePositives": 10, "falsePositives": 1, "falseNegatives": 1,
        "perClass": {
            "ball": {"groundTruth": 5, "predictions": 5, "truePositives": 5, "falsePositives": 0, "falseNegatives": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "ap50": 1.0, "meanIoU": 0.9},
            "referee": {"groundTruth": 5, "predictions": 6, "truePositives": 5, "falsePositives": 1, "falseNegatives": 0, "precision": 0.83, "recall": 1.0, "f1": 0.91, "ap50": 0.7, "meanIoU": 0.75}
        }
    }
    """.data(using: .utf8)!
    let decoded = try decoder.decode(BenchmarkMetrics.self, from: withPerClass)
    #expect(decoded.perClass["ball"]?.ap50 == 1.0)
    #expect(decoded.perClass["referee"]?.ap50 == 0.7)

    let withoutPerClass = """
    {
        "qualityScore": 88.5, "mAP50": 0.9, "precision": 0.95, "recall": 0.92, "f1": 0.93,
        "meanIoU": 0.8, "truePositives": 10, "falsePositives": 1, "falseNegatives": 1
    }
    """.data(using: .utf8)!
    let decodedWithoutPerClass = try decoder.decode(BenchmarkMetrics.self, from: withoutPerClass)
    #expect(decodedWithoutPerClass.perClass.isEmpty)
}

// isNewerVersion() backs the GitHub-releases update check: comparing tag_name
// (e.g. "v0.12.9") against AppState.appVersion (e.g. "0.12.8") decides whether the
// update banner shows. A wrong comparison here means either nagging the user about
// a version they already have, or silently missing a real update.
@Test func isNewerVersionComparesSemanticVersionsCorrectly() {
    #expect(isNewerVersion("0.12.9", than: "0.12.8"))
    #expect(isNewerVersion("v0.13.0", than: "0.12.8"))
    #expect(isNewerVersion("1.0.0", than: "0.12.8"))
    #expect(!isNewerVersion("0.12.8", than: "0.12.8"))
    #expect(!isNewerVersion("0.12.7", than: "0.12.8"))
    // A missing trailing component is treated as 0, not ignored.
    #expect(isNewerVersion("0.13", than: "0.12.9"))
    #expect(!isNewerVersion("0.12", than: "0.12.1"))
}

@Test func parseVersionComponentsStripsALeadingVPrefix() {
    #expect(parseVersionComponents("v0.12.8") == [0, 12, 8])
    #expect(parseVersionComponents("0.12.8") == [0, 12, 8])
    #expect(parseVersionComponents("V1.2") == [1, 2])
}

// FieldCorner must encode as a plain [x, y] array, not CGPoint's native
// {"x":...,"y":...} Codable form, because ml_worker.py's field_membership_checker
// reads each corner with `for fx, fy in corners` - a keyed object there would
// silently break unpacking instead of raising a clear error.
@Test func fieldCornerEncodesAsAPlainArrayPair() throws {
    let corner = FieldCorner(x: 0.1, y: 0.25)
    let data = try JSONEncoder().encode(corner)
    let json = try JSONSerialization.jsonObject(with: data) as? [Double]
    #expect(json == [0.1, 0.25])

    let decoded = try JSONDecoder().decode(FieldCorner.self, from: data)
    #expect(decoded == corner)
}

// Regression coverage: fieldGeometry must round-trip through the same
// save/load path as the rest of the project document (mirrors projectRoundTrip).
@Test func projectRoundTripPreservesFieldGeometry() throws {
    let root = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-field-geometry-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let store = ProjectStore(rootURL: root)
    var project = ProjectDocument(name: "Test", sport: .basketball, sourceFolder: "/tmp/videos")
    project.fieldGeometry = FieldGeometry(
        corners: [
            FieldCorner(x: 0.1, y: 0.1), FieldCorner(x: 0.9, y: 0.1),
            FieldCorner(x: 0.9, y: 0.9), FieldCorner(x: 0.1, y: 0.9),
        ],
        realWidth: 15.0,
        realLength: 28.0
    )

    try store.save(project)
    let loaded = try store.load()

    #expect(loaded.fieldGeometry?.corners.count == 4)
    #expect(loaded.fieldGeometry?.corners.first == FieldCorner(x: 0.1, y: 0.1))
    #expect(loaded.fieldGeometry?.realWidth == 15.0)
    #expect(loaded.fieldGeometry?.realLength == 28.0)
}

// selectedFrameID is a computed convenience view onto selectedFrameIDs (the
// sidebar List's real, Set-based selection, needed for native macOS Cmd/Shift-
// click multi-select). Every other call site (annotation editor, active-learning
// review) only cares about "the one currently displayed frame", so it must stay
// nil - not "the first of several" - whenever zero or multiple frames are selected.
@Test @MainActor func selectedFrameIDBridgesToTheUnderlyingSelectionSet() {
    let app = AppState()
    let first = UUID()
    let second = UUID()

    #expect(app.selectedFrameID == nil)

    app.selectedFrameID = first
    #expect(app.selectedFrameIDs == [first])
    #expect(app.selectedFrameID == first)

    app.selectedFrameIDs = [first, second]
    #expect(app.selectedFrameID == nil)

    app.selectedFrameID = nil
    #expect(app.selectedFrameIDs.isEmpty)
}

// Regression coverage for multi-select frame deletion: removeFrames() must
// remove every given frame in one batch, save once, and clear the selection.
@Test @MainActor func removeFramesDeletesEveryGivenFrameInOneBatch() throws {
    let folder = FileManager.default.temporaryDirectory
        .appending(path: "reco-trainer-remove-frames-test-\(UUID().uuidString)", directoryHint: .isDirectory)
    let app = AppState()
    app.selectedFolder = folder
    let frameA = FrameRecord(relativePath: "frames/a.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 0, width: 10, height: 10)
    let frameB = FrameRecord(relativePath: "frames/b.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 1, width: 10, height: 10)
    let frameC = FrameRecord(relativePath: "frames/c.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 2, width: 10, height: 10)
    app.project = ProjectDocument(name: "Test", sport: .basketball, sourceFolder: folder.path, frames: [frameA, frameB, frameC])
    app.selectedFrameIDs = [frameA.id, frameB.id]

    app.removeFrames([frameA.id, frameB.id])

    #expect(app.project?.frames.map(\.id) == [frameC.id])
    #expect(app.selectedFrameIDs.isEmpty)
}

// trainingCategories tracks an *exclusion* set rather than the selection itself,
// so a newly-annotated category (e.g. after auto-labeling referee mid-session)
// joins local training by default instead of needing to be ticked - only an
// explicit exclusion (e.g. "leave ball as it already is") sticks.
@Test @MainActor func trainingCategoriesExcludesOnlyWhatWasExplicitlyDeselected() {
    let app = AppState()
    let frame = FrameRecord(
        relativePath: "frames/one.jpg", videoID: "v1", videoName: "clip.mov", timestamp: 0,
        width: 10, height: 10,
        annotations: [
            BoxAnnotation(category: "ball", x: 1, y: 1, width: 2, height: 2),
            BoxAnnotation(category: "referee", x: 4, y: 4, width: 2, height: 2),
        ]
    )
    app.project = ProjectDocument(name: "Test", sport: .basketball, sourceFolder: "/tmp/videos", frames: [frame])

    #expect(app.trainingCategories == ["ball", "referee"])

    app.trainingExcludedCategories.insert("ball")
    #expect(app.trainingCategories == ["referee"])

    // A newly-annotated category joins training automatically, without
    // resetting the earlier exclusion.
    var updatedFrame = frame
    updatedFrame.annotations.append(BoxAnnotation(category: "hoop", x: 6, y: 6, width: 2, height: 2))
    app.project?.frames = [updatedFrame]
    #expect(app.trainingCategories == ["referee", "hoop"])
}
