// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "RecoTrainerMac",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "RecoTrainerMac", targets: ["RecoTrainerMac"])
    ],
    targets: [
        .executableTarget(
            name: "RecoTrainerMac",
            resources: [.copy("Resources")]
        ),
        .testTarget(
            name: "RecoTrainerMacTests",
            dependencies: ["RecoTrainerMac"]
        )
    ]
)
