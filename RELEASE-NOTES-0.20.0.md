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
