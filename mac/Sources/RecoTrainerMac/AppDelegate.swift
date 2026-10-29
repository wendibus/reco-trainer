import AppKit
import LauncherCore
import WebKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
    private let worker = WorkerController()
    private let store: PayloadStore
    private var window: NSWindow!
    private var webView: WKWebView!
    private var activePayload: Payload?
    private var updateCheckRunning = false

    private static let releasesPage = URL(string: "https://github.com/wendibus/reco-trainer/releases/latest")!

    override init() {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0.0.0"
        store = PayloadStore(
            installRoot: PayloadStore.defaultInstallRoot,
            // RECO_PAYLOAD_DIR lets `swift run` use a payload folder built by cross-platform/scripts/build-payload.sh.
            bundledRoot: ProcessInfo.processInfo.environment["RECO_PAYLOAD_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
                ?? Bundle.main.resourceURL?.appendingPathComponent("payload", isDirectory: true),
            launcherVersion: version
        )
        super.init()
    }

    // MARK: Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        buildWindow()
        NSApp.activate(ignoringOtherApps: true)
        Task { await launchWorkerAndLoad() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard worker.isRunning else { return .terminateNow }
        Task { @MainActor in
            var proceed = true
            if await worker.isBusy() {
                proceed = confirm(
                    L("Es läuft gerade eine Aufgabe (zum Beispiel Training). Beenden bricht sie ab.",
                      "A task is still running (for example training). Quitting will cancel it."),
                    ok: L("Trotzdem beenden", "Quit anyway"),
                    cancel: L("Weiterlaufen lassen", "Keep running")
                )
            }
            NSApp.reply(toApplicationShouldTerminate: proceed)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        worker.stop()
    }

    // MARK: Starting

    private func launchWorkerAndLoad() async {
        _ = try? store.seedBundledPayloadIfNeeded()
        var lastError: Error?
        for _ in 0..<3 {
            guard let payload = store.activePayload() else { break }
            do {
                try await worker.start(payload: payload)
                activePayload = payload
                loadInterface()
                store.prune(keeping: 2, protecting: [payload.version])
                Task { await checkForUpdates(userInitiated: false) }
                return
            } catch {
                lastError = error
                if case WorkerError.pythonNotFound = error { break }
                // This payload does not start: remember that and fall back to an older one.
                store.markBad(version: payload.version)
            }
        }
        showStartFailure(lastError)
    }

    private func loadInterface() {
        guard let url = URL(string: "\(worker.baseURL.absoluteString)/?launcher=1") else { return }
        webView.load(URLRequest(url: url))
    }

    private func showStartFailure(_ error: Error?) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = L("Reco Trainer konnte nicht starten", "Reco Trainer could not start")
        alert.informativeText = error?.localizedDescription
            ?? L("Es wurde keine startbare Programmversion gefunden. Bitte Reco Trainer neu installieren.",
                 "No runnable program version was found. Please reinstall Reco Trainer.")
        alert.addButton(withTitle: L("Beenden", "Quit"))
        alert.addButton(withTitle: L("Protokoll zeigen", "Show log"))
        if alert.runModal() == .alertSecondButtonReturn { showLog() }
        NSApp.terminate(nil)
    }

    // MARK: Updates

    @objc private func checkForUpdatesFromMenu() {
        Task { await checkForUpdates(userInitiated: true) }
    }

    private func checkForUpdates(userInitiated: Bool) async {
        guard !updateCheckRunning, let installed = activePayload?.version else { return }
        updateCheckRunning = true
        defer { updateCheckRunning = false }
        do {
            let release = try await Updater.fetchLatestRelease()
            switch UpdatePlanner.decide(release: release, installedVersion: installed) {
            case .upToDate:
                if userInitiated {
                    inform(L("Reco Trainer ist auf dem neuesten Stand (Version \(installed)).",
                             "Reco Trainer is up to date (version \(installed))."))
                }
            case .needsNewApp(let version, let page):
                let key = "notifiedNeedsNewApp"
                if !userInitiated && UserDefaults.standard.string(forKey: key) == version { return }
                UserDefaults.standard.set(version, forKey: key)
                if confirm(
                    L("Version \(version) ist erschienen und braucht eine neue App. Die Download-Seite öffnen?",
                      "Version \(version) is available and needs a new app download. Open the download page?"),
                    ok: L("Seite öffnen", "Open page"), cancel: L("Später", "Later")
                ) { NSWorkspace.shared.open(page) }
            case .available(let version, let zip, let signature):
                guard Updater.isConfigured else {
                    if userInitiated { inform(UpdaterError.notConfigured.localizedDescription) }
                    return
                }
                guard confirm(
                    L("Version \(version) ist verfügbar. Jetzt installieren?", "Version \(version) is available. Install now?"),
                    ok: L("Installieren", "Install"), cancel: L("Später", "Later")
                ) else { return }
                if await worker.isBusy() {
                    inform(L("Es läuft gerade eine Aufgabe. Bitte das Update nach deren Ende installieren.",
                             "A task is running. Please install the update after it has finished."))
                    return
                }
                window.title = L("Reco Trainer – installiere Update …", "Reco Trainer – installing update …")
                defer { window.title = "Reco Trainer" }
                let payload = try await Updater.install(zip: zip, signature: signature, into: store)
                await restart(with: payload)
            }
        } catch PayloadError.launcherTooOld(let required) {
            if confirm(
                L("Dieses Update braucht Reco Trainer \(required) oder neuer. Die Download-Seite öffnen?",
                  "This update needs Reco Trainer \(required) or newer. Open the download page?"),
                ok: L("Seite öffnen", "Open page"), cancel: L("Später", "Later")
            ) { NSWorkspace.shared.open(Self.releasesPage) }
        } catch {
            if userInitiated {
                inform(L("Die Update-Suche ist fehlgeschlagen: ", "Checking for updates failed: ") + error.localizedDescription)
            }
        }
    }

    private func restart(with payload: Payload) async {
        worker.stop()
        do {
            try await worker.start(payload: payload)
            activePayload = payload
            loadInterface()
        } catch {
            store.markBad(version: payload.version)
            await launchWorkerAndLoad()
        }
    }

    // MARK: Window and menu

    private func buildWindow() {
        let configuration = WKWebViewConfiguration()
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        if #available(macOS 13.3, *) { webView.isInspectable = true }

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1360, height: 900),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Reco Trainer"
        window.contentView = webView
        window.center()
        window.setFrameAutosaveName("RecoTrainerMainWindow")
        window.makeKeyAndOrderFront(nil)
        webView.loadHTMLString(
            "<body style='font-family:-apple-system;display:flex;height:100vh;align-items:center;justify-content:center;color:#667085'>"
                + "<div>\(L("Reco Trainer startet …", "Starting Reco Trainer …"))</div></body>",
            baseURL: nil
        )
    }

    private func buildMenu() {
        let main = NSMenu()

        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: L("Über Reco Trainer", "About Reco Trainer"),
                        action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        let updates = appMenu.addItem(withTitle: L("Nach Updates suchen …", "Check for Updates…"),
                                      action: #selector(checkForUpdatesFromMenu), keyEquivalent: "")
        updates.target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: L("Reco Trainer ausblenden", "Hide Reco Trainer"),
                        action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: L("Reco Trainer beenden", "Quit Reco Trainer"),
                        action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)

        let editItem = NSMenuItem()
        let editMenu = NSMenu(title: L("Bearbeiten", "Edit"))
        editMenu.addItem(withTitle: L("Widerrufen", "Undo"), action: Selector(("undo:")), keyEquivalent: "z")
        let redo = editMenu.addItem(withTitle: L("Wiederholen", "Redo"), action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: L("Ausschneiden", "Cut"), action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: L("Kopieren", "Copy"), action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: L("Einsetzen", "Paste"), action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: L("Alles auswählen", "Select All"), action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu
        main.addItem(editItem)

        let viewItem = NSMenuItem()
        let viewMenu = NSMenu(title: L("Ansicht", "View"))
        let reload = viewMenu.addItem(withTitle: L("Neu laden", "Reload"), action: #selector(reloadInterface), keyEquivalent: "r")
        reload.target = self
        let zoomIn = viewMenu.addItem(withTitle: L("Vergrößern", "Zoom In"), action: #selector(zoomIn), keyEquivalent: "+")
        zoomIn.target = self
        let zoomOut = viewMenu.addItem(withTitle: L("Verkleinern", "Zoom Out"), action: #selector(zoomOut), keyEquivalent: "-")
        zoomOut.target = self
        let zoomReset = viewMenu.addItem(withTitle: L("Originalgröße", "Actual Size"), action: #selector(zoomReset), keyEquivalent: "0")
        zoomReset.target = self
        viewMenu.addItem(.separator())
        viewMenu.addItem(withTitle: L("Vollbild", "Enter Full Screen"), action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
            .keyEquivalentModifierMask = [.command, .control]
        viewItem.submenu = viewMenu
        main.addItem(viewItem)

        let windowItem = NSMenuItem()
        let windowMenu = NSMenu(title: L("Fenster", "Window"))
        windowMenu.addItem(withTitle: L("Im Dock ablegen", "Minimize"), action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: L("Schließen", "Close"), action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        windowItem.submenu = windowMenu
        main.addItem(windowItem)
        NSApp.windowsMenu = windowMenu

        let helpItem = NSMenuItem()
        let helpMenu = NSMenu(title: L("Hilfe", "Help"))
        let log = helpMenu.addItem(withTitle: L("Protokoll im Finder zeigen", "Show Log in Finder"), action: #selector(showLogFromMenu), keyEquivalent: "")
        log.target = self
        let releases = helpMenu.addItem(withTitle: L("Releases auf GitHub", "Releases on GitHub"), action: #selector(openReleases), keyEquivalent: "")
        releases.target = self
        helpItem.submenu = helpMenu
        main.addItem(helpItem)
        NSApp.helpMenu = helpMenu

        NSApp.mainMenu = main
    }

    @objc private func reloadInterface() { webView.reload() }
    @objc private func zoomIn() { webView.pageZoom = min(webView.pageZoom + 0.1, 3) }
    @objc private func zoomOut() { webView.pageZoom = max(webView.pageZoom - 0.1, 0.5) }
    @objc private func zoomReset() { webView.pageZoom = 1 }
    @objc private func showLogFromMenu() { showLog() }
    @objc private func openReleases() { NSWorkspace.shared.open(Self.releasesPage) }

    private func showLog() {
        NSWorkspace.shared.activateFileViewerSelecting([WorkerController.logURL])
    }

    // MARK: Dialog helpers

    private func inform(_ message: String) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    private func confirm(_ message: String, ok: String, cancel: String) -> Bool {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: ok)
        alert.addButton(withTitle: cancel)
        return alert.runModal() == .alertFirstButtonReturn
    }

    // MARK: WKNavigationDelegate

    /// Only the local interface may load in this window; everything else opens in the default browser.
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction) async -> WKNavigationActionPolicy {
        guard let url = navigationAction.request.url else { return .cancel }
        if url.scheme == "about" { return .allow }
        if (url.host == "127.0.0.1" || url.host == "localhost") && url.port == worker.port { return .allow }
        if url.scheme == "http" || url.scheme == "https" || url.scheme == "mailto" {
            NSWorkspace.shared.open(url)
        }
        return .cancel
    }

    // MARK: WKUIDelegate - the web interface uses confirm() and prompt(), which WKWebView ignores unless handled here

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url { NSWorkspace.shared.open(url) }
        return nil
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo) async {
        inform(message)
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo) async -> Bool {
        confirm(message, ok: "OK", cancel: L("Abbrechen", "Cancel"))
    }

    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
                 defaultText: String?, initiatedByFrame frame: WKFrameInfo) async -> String? {
        let alert = NSAlert()
        alert.messageText = prompt
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 320, height: 24))
        field.stringValue = defaultText ?? ""
        alert.accessoryView = field
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: L("Abbrechen", "Cancel"))
        alert.window.initialFirstResponder = field
        return alert.runModal() == .alertFirstButtonReturn ? field.stringValue : nil
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo) async -> [URL]? {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        return panel.runModal() == .OK ? panel.urls : nil
    }
}
