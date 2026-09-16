import AppKit
import SwiftUI

/// Read-only playback of a ball-tracking simulation: cycles through the
/// throwaway-extracted frames at the clip's own frame rate, drawing the
/// resolved ball marker on each one. Reuses AnnotationEditor's aspect-fit
/// image layout idea, stripped of every drag/edit gesture since this view
/// never modifies anything.
struct BallTrackingPlayer: View {
    let simulation: BallTrackingSimulation
    let fieldGeometry: FieldGeometry?
    let language: AppLanguage
    @Binding var lookaheadFrames: Int

    @State private var currentIndex = 0
    @State private var isPlaying = true
    @State private var playbackTask: Task<Void, Never>?

    private var resolved: [ResolvedBallPosition] {
        BallTrackingResolver.resolve(detections: simulation.detections, lookaheadFrames: lookaheadFrames)
    }

    /// The field boundary's four fractional corners give an image-space crop
    /// rectangle directly - no homography needed, that's only for real-world
    /// distance (see field_membership_checker in ml_worker.py).
    private var cropFraction: CGRect? {
        guard let corners = fieldGeometry?.corners, corners.count == 4 else { return nil }
        let xs = corners.map(\.x), ys = corners.map(\.y)
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else { return nil }
        let margin = 0.05
        let x0 = max(0, minX - margin), y0 = max(0, minY - margin)
        let x1 = min(1, maxX + margin), y1 = min(1, maxY + margin)
        guard x1 > x0, y1 > y0 else { return nil }
        return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }

    var body: some View {
        VStack(spacing: 12) {
            if simulation.frames.indices.contains(currentIndex) {
                let frame = simulation.frames[currentIndex]
                let position = resolved.indices.contains(currentIndex) ? resolved[currentIndex] : nil
                GeometryReader { geometry in
                    let container = CGRect(origin: .zero, size: geometry.size)
                    let imageSize = CGSize(width: frame.width, height: frame.height)
                    let fitted = fittedImageRect(imageSize: imageSize, cropFraction: cropFraction, in: container)

                    ZStack(alignment: .topLeading) {
                        if let image = NSImage(contentsOf: simulation.store.frameURL(for: frame)) {
                            Image(nsImage: image)
                                .resizable()
                                .frame(width: fitted.width, height: fitted.height)
                                .position(x: fitted.midX, y: fitted.midY)
                        }
                        Canvas { context, _ in
                            guard let position, let x = position.x, let y = position.y else { return }
                            let point = CGPoint(x: fitted.minX + x / Double(frame.width) * fitted.width, y: fitted.minY + y / Double(frame.height) * fitted.height)
                            let radius: Double = position.state == .detected ? 9 : 7
                            let dot = CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)
                            switch position.state {
                            case .detected:
                                context.fill(Path(ellipseIn: dot), with: .color(.green))
                            case .interpolated:
                                context.stroke(Path(ellipseIn: dot), with: .color(.orange), style: StrokeStyle(lineWidth: 2.5, dash: [4, 3]))
                            case .coasting:
                                context.stroke(Path(ellipseIn: dot), with: .color(.yellow), style: StrokeStyle(lineWidth: 2.5, dash: [2, 4]))
                            case .lost:
                                break
                            }
                        }
                    }
                    .frame(width: container.width, height: container.height)
                    .clipped()
                }
                .frame(minHeight: 360)
                .background(.black.opacity(0.85))
                .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                ContentUnavailableView(
                    language.text("Keine Bilder", "No images"),
                    systemImage: "photo.badge.exclamationmark"
                )
                .frame(minHeight: 360)
            }

            legend

            HStack(spacing: 10) {
                Button {
                    isPlaying.toggle()
                    if isPlaying { startPlayback() } else { playbackTask?.cancel() }
                } label: {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                }
                .buttonStyle(.bordered)

                Slider(
                    value: Binding(
                        get: { Double(currentIndex) },
                        set: { currentIndex = min(max(0, Int($0)), max(0, simulation.frames.count - 1)) }
                    ),
                    in: 0...Double(max(0, simulation.frames.count - 1))
                )
                .onChange(of: currentIndex) { _, _ in }

                Text("\(currentIndex + 1) / \(simulation.frames.count)")
                    .font(.caption.monospacedDigit())
                    .frame(width: 70, alignment: .trailing)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(language.text(
                        "Wie weit direkt in die Zukunft schauen",
                        "How far to look directly into the future",
                        "Hasta dónde mirar directamente al futuro",
                        "Jusqu’où regarder directement dans le futur"
                    ))
                    Spacer()
                    Text(lookaheadSecondsLabel).monospacedDigit().foregroundStyle(.secondary)
                }
                .font(.caption)
                Slider(value: Binding(
                    get: { Double(lookaheadFrames) },
                    set: { lookaheadFrames = Int($0) }
                ), in: 0...Double(max(1, Int(simulation.fps * 5))), step: 1)
            }
        }
        .onAppear { startPlayback() }
        .onDisappear { playbackTask?.cancel() }
    }

    private var lookaheadSecondsLabel: String {
        let seconds = simulation.fps > 0 ? Double(lookaheadFrames) / simulation.fps : 0
        return String(format: "%.1fs (%d %@)", seconds, lookaheadFrames, language.text("Bilder", "frames", "imágenes", "images"))
    }

    private var legend: some View {
        HStack(spacing: 16) {
            legendItem(color: .green, label: language.text("Erkannt", "Detected", "Detectado", "Détecté"))
            legendItem(color: .orange, label: language.text("Interpoliert", "Interpolated", "Interpolado", "Interpolé"))
            legendItem(color: .yellow, label: language.text("Gehalten", "Held", "Mantenido", "Maintenu"))
            legendItem(color: .secondary, label: language.text("Verloren", "Lost", "Perdido", "Perdu"))
        }
        .font(.caption2)
    }

    private func legendItem(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label).foregroundStyle(.secondary)
        }
    }

    private func startPlayback() {
        playbackTask?.cancel()
        guard isPlaying, simulation.frames.count > 1 else { return }
        let interval = simulation.fps > 0 ? 1.0 / simulation.fps : 1.0 / 12.0
        playbackTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(interval))
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    currentIndex = (currentIndex + 1) % simulation.frames.count
                }
            }
        }
    }

    /// Positions/scales the full image so that cropFraction (if given, a
    /// fractional 0...1 rect within the image) fills `container` - the rest
    /// of the image extends beyond the container and is clipped by the
    /// caller. Falls back to a plain aspect-fit of the whole image when no
    /// crop is set.
    private func fittedImageRect(imageSize: CGSize, cropFraction: CGRect?, in container: CGRect) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return container }
        guard let cropFraction, cropFraction.width > 0, cropFraction.height > 0 else {
            let scale = min(container.width / imageSize.width, container.height / imageSize.height)
            let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
            return CGRect(x: container.midX - size.width / 2, y: container.midY - size.height / 2, width: size.width, height: size.height)
        }
        let cropSize = CGSize(width: imageSize.width * cropFraction.width, height: imageSize.height * cropFraction.height)
        let scale = min(container.width / cropSize.width, container.height / cropSize.height)
        let fullSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let cropOrigin = CGPoint(x: imageSize.width * cropFraction.minX * scale, y: imageSize.height * cropFraction.minY * scale)
        let cropFittedSize = CGSize(width: cropSize.width * scale, height: cropSize.height * scale)
        let cropFittedOrigin = CGPoint(x: container.midX - cropFittedSize.width / 2, y: container.midY - cropFittedSize.height / 2)
        return CGRect(
            x: cropFittedOrigin.x - cropOrigin.x,
            y: cropFittedOrigin.y - cropOrigin.y,
            width: fullSize.width,
            height: fullSize.height
        )
    }
}
