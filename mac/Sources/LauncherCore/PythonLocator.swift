import Foundation

public enum PythonLocator {
    /// Preferred interpreters for running the worker, newest first. The worker itself runs on any
    /// Python 3; "Set up ML" later looks for a 3.10+ interpreter on its own.
    public static func candidates() -> [String] {
        var result: [String] = []
        for minor in [13, 12, 11, 10] {
            result.append("/opt/homebrew/bin/python3.\(minor)")
            result.append("/usr/local/bin/python3.\(minor)")
            result.append("/Library/Frameworks/Python.framework/Versions/3.\(minor)/bin/python3")
        }
        result.append("/opt/homebrew/bin/python3")
        result.append("/usr/local/bin/python3")
        result.append("/usr/bin/python3")
        return result
    }

    public static func firstAvailable(
        isExecutable: (String) -> Bool = { FileManager.default.isExecutableFile(atPath: $0) },
        commandLineToolsInstalled: () -> Bool = { true }
    ) -> String? {
        for path in candidates() where isExecutable(path) {
            // /usr/bin/python3 is only a stub that opens an installer prompt without the Xcode command line tools.
            if path == "/usr/bin/python3" && !commandLineToolsInstalled() { continue }
            return path
        }
        return nil
    }
}
