import Darwin
import Foundation
import LauncherCore

enum WorkerError: Error, LocalizedError {
    case pythonNotFound
    case exited(Int32)
    case timeout

    var errorDescription: String? {
        switch self {
        case .pythonNotFound:
            return L(
                "Python 3 wurde nicht gefunden. Bitte Python 3.12 installieren (zum Beispiel mit „brew install python@3.12“ oder von python.org) und Reco Trainer neu starten.",
                "Python 3 was not found. Please install Python 3.12 (for example with “brew install python@3.12” or from python.org) and restart Reco Trainer."
            )
        case .exited(let status):
            return L("Der Worker wurde unerwartet beendet (Status \(status)).", "The worker exited unexpectedly (status \(status)).")
        case .timeout:
            return L("Der Worker hat sich nicht rechtzeitig gemeldet.", "The worker did not respond in time.")
        }
    }
}

/// Starts and supervises `local_worker.py`, which serves the web interface and does all the work.
final class WorkerController {
    static let preferredPort = 8766
    /// Keep identical to LAUNCH_SNIPPET in cross-platform/tests/test_launcher_contract.py.
    /// The worker must lead its own process group, so quitting can stop it together with everything
    /// it spawned (training processes) in one go. Process already starts the child as a group leader,
    /// and setsid() would then fail with EPERM - so it is only called when that is not the case.
    static let launchSnippet = "import os, sys; os.getpgrp() == os.getpid() or os.setsid(); os.execv(sys.executable, [sys.executable] + sys.argv[1:])"

    private(set) var port = WorkerController.preferredPort
    private var process: Process?

    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }
    var isRunning: Bool { process?.isRunning == true }

    static var logURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Reco Trainer/worker.log")
    }

    func start(payload: Payload) async throws {
        stop()
        guard let python = PythonLocator.firstAvailable(commandLineToolsInstalled: Self.commandLineToolsInstalled) else {
            throw WorkerError.pythonNotFound
        }
        port = Self.firstFreePort(startingAt: Self.preferredPort)

        let logURL = Self.logURL
        try FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? FileManager.default.attributesOfItem(atPath: logURL.path)[.size]) as? Int, size > 5_000_000 {
            try? FileManager.default.removeItem(at: logURL)
        }
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let log = try FileHandle(forWritingTo: logURL)
        try log.seekToEnd()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = ["-c", Self.launchSnippet, payload.workerScript.path, "--port", String(port)]
        process.currentDirectoryURL = payload.directory
        var environment = ProcessInfo.processInfo.environment
        // GUI apps start with a minimal PATH; ffmpeg and Homebrew Python live here.
        environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if let extractor = Bundle.main.url(forAuxiliaryExecutable: "reco-frame-extractor") {
            environment["RECO_FRAME_EXTRACTOR"] = extractor.path
        }
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = log
        process.standardError = log
        log.write(Data("\n--- Reco Trainer launcher: starting payload \(payload.version) on port \(port) with \(python) ---\n".utf8))
        try process.run()
        self.process = process

        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            if !process.isRunning {
                self.process = nil
                throw WorkerError.exited(process.terminationStatus)
            }
            if await Self.isHealthy(port: port) { return }
            try await Task.sleep(nanoseconds: 300_000_000)
        }
        stop()
        throw WorkerError.timeout
    }

    /// Stops the worker and its whole process group (SIGTERM, then SIGKILL after a few seconds).
    func stop() {
        guard let process else { return }
        self.process = nil
        guard process.isRunning else { return }
        let pid = process.processIdentifier
        kill(-pid, SIGTERM)
        let deadline = Date().addingTimeInterval(4)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.1) }
        if process.isRunning { kill(-pid, SIGKILL) }
    }

    /// True while the worker reports a running operation (preparing, auto-labelling, training ...).
    func isBusy() async -> Bool {
        guard let object = await Self.fetchStatus(port: port) else { return false }
        return (object["busy"] as? Bool) == true
    }

    // MARK: Helpers

    private static func fetchStatus(port: Int) async -> [String: Any]? {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/status")!)
        request.timeoutInterval = 2
        guard
            let (data, response) = try? await URLSession.shared.data(for: request),
            (response as? HTTPURLResponse)?.statusCode == 200
        else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func isHealthy(port: Int) async -> Bool {
        await fetchStatus(port: port) != nil
    }

    private static func commandLineToolsInstalled() -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcode-select")
        process.arguments = ["-p"]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return false }
        process.waitUntilExit()
        return process.terminationStatus == 0
    }

    static func firstFreePort(startingAt start: Int) -> Int {
        for port in start..<(start + 40) where isPortFree(UInt16(port)) { return port }
        return start
    }

    private static func isPortFree(_ port: UInt16) -> Bool {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { return false }
        defer { close(descriptor) }
        // Like the worker's ThreadingHTTPServer (allow_reuse_address): connections of the previous
        // run in TIME_WAIT must not count as "busy", or the port - and with it the web view's
        // per-origin settings (language, onboarding) - would change on a quick restart.
        var reuse: Int32 = 1
        setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        return withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }
}
