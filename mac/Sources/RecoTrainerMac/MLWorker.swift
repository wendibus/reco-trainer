import Foundation

private final class LockedOutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ""

    func append(_ chunk: String) {
        lock.lock()
        value += chunk
        lock.unlock()
    }

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

enum MLWorkerError: LocalizedError {
    case missingResource
    case pythonNotFound
    case failed(Int32, String)

    var errorDescription: String? {
        switch self {
        case .missingResource: "Der eingebettete ML-Worker fehlt."
        case .pythonNotFound: "Python 3.11 oder 3.12 wurde nicht gefunden."
        case .failed(let code, let output): "ML-Worker beendet mit Code \(code):\n\(output)"
        }
    }
}

struct MLWorker: Sendable {
    let projectRoot: URL

    private var workerURL: URL {
        get throws {
            if let resources = Bundle.main.resourceURL {
                let packaged = resources
                    .appending(path: "RecoTrainerMac_RecoTrainerMac.bundle", directoryHint: .isDirectory)
                    .appending(path: "Resources", directoryHint: .isDirectory)
                    .appending(path: "ml_worker.py")
                if FileManager.default.fileExists(atPath: packaged.path) {
                    return packaged
                }
            }
            guard let url = Bundle.module.url(forResource: "ml_worker", withExtension: "py", subdirectory: "Resources")
                ?? Bundle.module.url(forResource: "ml_worker", withExtension: "py") else {
                throw MLWorkerError.missingResource
            }
            return url
        }
    }

    var venvURL: URL { projectRoot.appending(path: ".runtime/venv", directoryHint: .isDirectory) }
    var venvPythonURL: URL { venvURL.appending(path: "bin/python3") }

    func doctor() async throws -> HardwareStatus {
        let output = try await runPython(arguments: [try workerURL.path, "doctor"], preferVenv: true)
        let data = Data(output.utf8)
        return try JSONDecoder().decode(HardwareStatus.self, from: data)
    }

    func prepareEnvironment(onOutput: @escaping @Sendable (String) async -> Void) async throws {
        let manager = FileManager.default
        try manager.createDirectory(at: venvURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !manager.fileExists(atPath: venvPythonURL.path) {
            let python = try systemPython()
            _ = try await run(executable: python, arguments: ["-m", "venv", venvURL.path], onOutput: onOutput)
        }
        _ = try await run(
            executable: venvPythonURL,
            arguments: ["-m", "pip", "install", "--upgrade", "pip", "rfdetr[train,onnx,coreml]>=1.9.0", "onnxruntime"],
            onOutput: onOutput
        )
    }

    func autoLabel(
        modelSize: ModelSize,
        categories: [String],
        threshold: Double,
        candidateOnly: Bool = false,
        language: AppLanguage,
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws {
        var arguments = [
                try workerURL.path, "autolabel", "--project", projectRoot.path,
                "--model", modelSize.rawValue, "--category",
            ] + categories + [
                "--threshold", String(threshold), "--language", language.rawValue
            ]
        if candidateOnly { arguments.append("--candidate-only") }
        _ = try await runPython(
            arguments: arguments,
            preferVenv: true,
            onOutput: onOutput
        )
    }

    func train(
        modelSize: ModelSize,
        epochs: Int,
        categories: [String]? = nil,
        language: AppLanguage,
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws {
        var arguments = [
            try workerURL.path, "train", "--project", projectRoot.path,
            "--model", modelSize.rawValue, "--epochs", String(epochs),
        ]
        if let categories, !categories.isEmpty {
            arguments += ["--category"] + categories
        }
        arguments += ["--language", language.rawValue]
        _ = try await runPython(
            arguments: arguments,
            preferVenv: true,
            onOutput: onOutput
        )
    }

    func refineBoxes(
        language: AppLanguage,
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws {
        _ = try await runPython(
            arguments: [
                try workerURL.path, "refine-boxes", "--project", projectRoot.path,
                "--language", language.rawValue
            ],
            preferVenv: true,
            onOutput: onOutput
        )
    }

    func exportCPU(
        modelSize: ModelSize,
        language: AppLanguage,
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws {
        _ = try await runPython(
            arguments: [
                try workerURL.path, "export", "--project", projectRoot.path,
                "--model", modelSize.rawValue, "--format", "onnx", "--language", language.rawValue
            ],
            preferVenv: true,
            onOutput: onOutput
        )
    }

    func exportCoreML(
        modelSize: ModelSize,
        language: AppLanguage,
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws {
        _ = try await runPython(
            arguments: [
                try workerURL.path, "export", "--project", projectRoot.path,
                "--model", modelSize.rawValue, "--format", "coreml", "--language", language.rawValue
            ],
            preferVenv: true,
            onOutput: onOutput
        )
    }

    func packageModel(
        modelSize: ModelSize,
        name: String,
        language: AppLanguage,
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws {
        _ = try await runPython(
            arguments: [
                try workerURL.path, "package", "--project", projectRoot.path,
                "--model", modelSize.rawValue, "--name", name, "--language", language.rawValue
            ],
            preferVenv: true,
            onOutput: onOutput
        )
    }

    func benchmark(
        threshold: Double,
        language: AppLanguage,
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws {
        _ = try await runPython(
            arguments: [
                try workerURL.path, "benchmark", "--project", projectRoot.path,
                "--threshold", String(threshold), "--language", language.rawValue
            ],
            preferVenv: true,
            onOutput: onOutput
        )
    }

    func simulateBallTracking(
        framesDirectory: URL,
        modelSize: ModelSize,
        threshold: Double,
        language: AppLanguage
    ) async throws -> BallSimulationResponse {
        let output = try await runPython(
            arguments: [
                try workerURL.path, "simulate-ball-tracking", "--project", projectRoot.path,
                "--frames-dir", framesDirectory.path, "--model", modelSize.rawValue,
                "--threshold", String(threshold), "--language", language.rawValue
            ],
            preferVenv: true
        )
        do {
            return try Self.decodeLastJSONLine(BallSimulationResponse.self, from: output)
        } catch {
            throw MLWorkerError.failed(-1, output)
        }
    }

    /// Unlike installModelPackage/activateModel below (system Python, no ML
    /// libraries touched), simulateBallTracking loads the real RF-DETR/PyTorch
    /// stack - which can print its own warnings/notices to stderr during model
    /// loading. run(...) merges stdout+stderr into one stream (by design, so
    /// streaming actions can show everything), so decoding the *entire*
    /// captured output as one JSON blob broke the moment any such stray line
    /// appeared before the final print(json.dumps(...)) in
    /// simulate_ball_tracking() - surfacing as a generic "data isn't in the
    /// correct format" error with no indication why. Only the last non-empty
    /// line is guaranteed to be that final JSON print. Static and pure (no I/O)
    /// so it's unit-testable independent of actually spawning a process.
    static func decodeLastJSONLine<T: Decodable>(_ type: T.Type, from output: String) throws -> T {
        guard let lastLine = output.split(separator: "\n", omittingEmptySubsequences: true).last else {
            throw MLWorkerError.failed(-1, output)
        }
        return try JSONDecoder().decode(type, from: Data(lastLine.utf8))
    }

    func installModelPackage(
        fileURL: URL,
        language: AppLanguage
    ) async throws -> InstalledModelResponse {
        let output = try await runPython(
            arguments: [
                try workerURL.path, "install-package", "--project", projectRoot.path,
                "--file", fileURL.path, "--language", language.rawValue
            ],
            preferVenv: false
        )
        return try JSONDecoder().decode(InstalledModelResponse.self, from: Data(output.utf8))
    }

    func activateModel(packageID: String, language: AppLanguage) async throws -> ActivatedModelResponse {
        let output = try await runPython(
            arguments: [try workerURL.path, "activate-model", "--project", projectRoot.path, "--package-id", packageID, "--language", language.rawValue],
            preferVenv: false
        )
        return try JSONDecoder().decode(ActivatedModelResponse.self, from: Data(output.utf8))
    }

    func renameModel(packageID: String, name: String, language: AppLanguage) async throws {
        _ = try await runPython(
            arguments: [try workerURL.path, "rename-model", "--project", projectRoot.path, "--package-id", packageID, "--name", name, "--language", language.rawValue],
            preferVenv: false
        )
    }

    func deleteModel(packageID: String, language: AppLanguage) async throws {
        _ = try await runPython(
            arguments: [try workerURL.path, "delete-model", "--project", projectRoot.path, "--package-id", packageID, "--language", language.rawValue],
            preferVenv: false
        )
    }

    private func runPython(
        arguments: [String],
        preferVenv: Bool,
        onOutput: @escaping @Sendable (String) async -> Void = { _ in }
    ) async throws -> String {
        let executable: URL
        if preferVenv, FileManager.default.fileExists(atPath: venvPythonURL.path) {
            executable = venvPythonURL
        } else {
            executable = try systemPython()
        }
        return try await run(executable: executable, arguments: arguments, onOutput: onOutput)
    }

    private func systemPython() throws -> URL {
        let candidates = [
            "/opt/homebrew/bin/python3.12",
            "/usr/local/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/usr/local/bin/python3.11",
            "/usr/bin/python3"
        ]
        guard let path = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw MLWorkerError.pythonNotFound
        }
        return URL(filePath: path)
    }

    private func run(
        executable: URL,
        arguments: [String],
        onOutput: @escaping @Sendable (String) async -> Void
    ) async throws -> String {
        try await Task.detached {
            let process = Process()
            let pipe = Pipe()
            process.executableURL = executable
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe
            process.environment = ProcessInfo.processInfo.environment.merging([
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "RF_HOME": self.projectRoot.appending(path: ".runtime/models").path
            ]) { _, new in new }

            let outputBuffer = LockedOutputBuffer()
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty, let chunk = String(data: data, encoding: .utf8) else { return }
                outputBuffer.append(chunk)
                Task { await onOutput(chunk) }
            }

            try process.run()
            process.waitUntilExit()
            pipe.fileHandleForReading.readabilityHandler = nil
            let remainder = pipe.fileHandleForReading.readDataToEndOfFile()
            if let chunk = String(data: remainder, encoding: .utf8), !chunk.isEmpty {
                outputBuffer.append(chunk)
                await onOutput(chunk)
            }
            let completeOutput = outputBuffer.snapshot()
            guard process.terminationStatus == 0 else {
                throw MLWorkerError.failed(process.terminationStatus, completeOutput)
            }
            return completeOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        }.value
    }
}
