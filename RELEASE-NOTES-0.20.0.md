# Reco Trainer 0.20.0

## Simpler to install and run

The web interface is now built ahead of time and served by the Python worker itself. **Windows, Linux and Docker users no longer need Node.js or npm.** Python 3.10+ (and FFmpeg) is all that is required.

- The interface now opens at <http://localhost:8766/> (one port instead of two). The old Vite port 8765 is only used by `npm run dev` for development.
- Start scripts on macOS, Windows and Linux just start the worker and open the browser. Nothing is installed on first start any more, so it starts in about two seconds.
- The Docker image builds the interface in a first stage and runs only Python; `docker compose` publishes only port 8766 on `127.0.0.1`.
- The interface toolchain dropped Next.js, vinext, Cloudflare and Wrangler for plain Vite and React. The look and behavior of the interface are unchanged.

## For developers

- `npm run dev` starts the Vite dev server on 8765 against a worker on 8766; `npm run build` writes `ui/`; `npm run typecheck` replaces the old lint script.
- `RECO_UI_DIR` points the worker at a different built interface folder.
- CI builds and type-checks the interface in addition to the Python tests.

## Mac: the app is now a launcher with built-in updates

- The Mac app shows the same web interface as every other platform, so features and look are identical everywhere. The separate SwiftUI interface is gone.
- **Updates install themselves:** on start (or via "Check for Updates…") the app offers new interface/worker versions from GitHub, verifies an Ed25519 signature, installs and restarts the worker. If a new version does not start, the previous one is used automatically. A new app download is only needed when the launcher itself changes.
- A precompiled frame extractor is bundled, so videos can be prepared without FFmpeg or Xcode tools.
- Existing projects keep working unchanged (same `.reco-training` folders).
- Python 3 is still required (3.12 recommended). Chrome, Node.js and Xcode are not.
