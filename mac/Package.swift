// swift-tools-version: 6.0
import PackageDescription

// The Mac app is a thin launcher: a native window around the web interface, plus the
// code that starts the Python worker and installs signed updates of the interface and
// workers ("payload"). Everything else lives in cross-platform/.
let package = Package(
    name: "RecoTrainerMac",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "RecoTrainerMac", targets: ["RecoTrainerMac"])
    ],
    targets: [
        // Foundation-only logic (versions, payload store, update planning, signatures) - unit tested.
        .target(name: "LauncherCore"),
        // AppKit/WebKit shell and worker process management.
        .executableTarget(name: "RecoTrainerMac", dependencies: ["LauncherCore"]),
        .testTarget(name: "LauncherCoreTests", dependencies: ["LauncherCore"]),
    ],
    swiftLanguageModes: [.v5]
)
