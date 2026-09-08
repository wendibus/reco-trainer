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
