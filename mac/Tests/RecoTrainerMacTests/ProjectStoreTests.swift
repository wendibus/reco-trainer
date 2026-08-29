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
