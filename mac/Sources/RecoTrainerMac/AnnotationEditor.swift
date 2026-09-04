import AppKit
import SwiftUI

struct AnnotationEditor: View {
    let imageURL: URL
    let frame: FrameRecord
    let selectedCategory: String
    let language: AppLanguage
    let onChange: ([BoxAnnotation]) -> Void

    @State private var dragStart: CGPoint?
    @State private var dragCurrent: CGPoint?
    @State private var zoom = 1.0
    @GestureState private var pinchScale = 1.0
    @State private var offset: CGSize = .zero
    @State private var panStartOffset: CGSize?
    @State private var isPanning = false
    @State private var undoStack: [[BoxAnnotation]] = []
    @State private var redoStack: [[BoxAnnotation]] = []

    init(
        imageURL: URL,
        frame: FrameRecord,
        selectedCategory: String,
        language: AppLanguage,
        onChange: @escaping ([BoxAnnotation]) -> Void
    ) {
        self.imageURL = imageURL
        self.frame = frame
        self.selectedCategory = selectedCategory
        self.language = language
        self.onChange = onChange
    }

    var body: some View {
        VStack(spacing: 10) {
            GeometryReader { geometry in
                let container = CGRect(origin: .zero, size: geometry.size)
                let imageSize = CGSize(width: frame.width, height: frame.height)
                let baseRect = aspectFit(imageSize: imageSize, in: container)
                let effectiveZoom = min(max(zoom * pinchScale, 1), 8)
                let fitted = zoomedRect(baseRect: baseRect, container: container, scale: effectiveZoom, offset: offset)

                ZStack(alignment: .topLeading) {
                    if let image = NSImage(contentsOf: imageURL) {
                        Image(nsImage: image)
                            .resizable()
                            .frame(width: fitted.width, height: fitted.height)
                            .position(x: fitted.midX, y: fitted.midY)
                    } else {
                        ContentUnavailableView(
                            language.text("Frame fehlt", "Frame is missing"),
                            systemImage: "photo.badge.exclamationmark"
                        )
                    }

                    Canvas { context, _ in
                        for annotation in frame.annotations {
                            let rect = screenRect(for: annotation, imageRect: fitted)
                            let color = color(for: annotation.category)
                            context.stroke(Path(rect), with: .color(color), lineWidth: 3)
                            context.draw(
                                Text(language.category(annotation.category)).font(.caption.bold()).foregroundStyle(.white),
                                at: CGPoint(x: rect.minX + 5, y: max(rect.minY - 10, fitted.minY + 10)),
                                anchor: .leading
                            )
                        }
                        if let dragStart, let dragCurrent {
                            let rect = normalizedRect(from: dragStart, to: dragCurrent)
                            context.stroke(Path(rect), with: .color(.yellow), style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 3, coordinateSpace: .local)
                            .onChanged { value in
                                if isPanning {
                                    let start = panStartOffset ?? offset
                                    if panStartOffset == nil { panStartOffset = offset }
                                    offset = clampedOffset(
                                        CGSize(
                                            width: start.width + value.translation.width,
                                            height: start.height + value.translation.height
                                        ),
                                        baseRect: baseRect,
                                        container: container,
                                        scale: effectiveZoom
                                    )
                                } else {
                                    dragStart = clamp(value.startLocation, to: fitted)
                                    dragCurrent = clamp(value.location, to: fitted)
                                }
                            }
                            .onEnded { value in
                                if isPanning {
                                    panStartOffset = nil
                                    return
                                }
                                let start = clamp(value.startLocation, to: fitted)
                                let end = clamp(value.location, to: fitted)
                                let rect = normalizedRect(from: start, to: end)
                                guard rect.width >= 5, rect.height >= 5, fitted.contains(start) else {
                                    dragStart = nil
                                    dragCurrent = nil
                                    return
                                }
                                let scaleX = Double(frame.width) / fitted.width
                                let scaleY = Double(frame.height) / fitted.height
                                var updated = frame.annotations
                                updated.append(BoxAnnotation(
                                    category: selectedCategory,
                                    x: Double(rect.minX - fitted.minX) * scaleX,
                                    y: Double(rect.minY - fitted.minY) * scaleY,
                                    width: Double(rect.width) * scaleX,
                                    height: Double(rect.height) * scaleY
                                ))
                                dragStart = nil
                                dragCurrent = nil
                                commit(updated)
                            }
                    )
                    .simultaneousGesture(
                        MagnifyGesture()
                            .updating($pinchScale) { value, state, _ in
                                state = value.magnification
                            }
                            .onEnded { value in
                                zoom = min(max(zoom * value.magnification, 1), 8)
                                offset = clampedOffset(
                                    offset,
                                    baseRect: baseRect,
                                    container: container,
                                    scale: zoom
                                )
                                if zoom == 1 { offset = .zero }
                            }
                    )
                }
                .clipped()
            }
            .background(.black.opacity(0.92), in: RoundedRectangle(cornerRadius: 12))

            HStack {
                Button {
                    isPanning.toggle()
                } label: {
                    Label(
                        isPanning
                            ? language.text("Markieren", "Annotate")
                            : language.text("Verschieben", "Pan"),
                        systemImage: isPanning ? "rectangle.dashed" : "hand.draw"
                    )
                }
                .buttonStyle(.borderedProminent)
                .tint(isPanning ? .orange : .blue)

                Label(
                    isPanning
                        ? language.text("Bild ziehen; Pinch zum Zoomen", "Drag image; pinch to zoom")
                        : language.text("Rahmen aufziehen; Pinch zum Zoomen", "Draw a box; pinch to zoom"),
                    systemImage: isPanning ? "hand.draw" : "rectangle.dashed"
                )
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    zoom = max(1, zoom / 1.5)
                    offset = .zero
                } label: {
                    Image(systemName: "minus.magnifyingglass")
                }
                .disabled(zoom <= 1)
                Text(String(format: "%.1f×", zoom))
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 42)
                Button {
                    zoom = min(zoom * 1.5, 8)
                } label: {
                    Image(systemName: "plus.magnifyingglass")
                }
                Button(language.text("Zurücksetzen", "Reset")) {
                    zoom = 1
                    offset = .zero
                }
                .disabled(zoom == 1 && offset == .zero)
                Divider().frame(height: 22)
                Button {
                    undo()
                } label: {
                    Label(language.text("Rückgängig", "Undo", "Deshacer", "Annuler"), systemImage: "arrow.uturn.backward")
                }
                .keyboardShortcut("z", modifiers: .command)
                .disabled(undoStack.isEmpty)
                Button {
                    redo()
                } label: {
                    Label(language.text("Wiederholen", "Redo", "Rehacer", "Rétablir"), systemImage: "arrow.uturn.forward")
                }
                .keyboardShortcut("z", modifiers: [.command, .shift])
                .disabled(redoStack.isEmpty)
                Button(language.text("Letzte entfernen", "Remove last")) {
                    guard !frame.annotations.isEmpty else { return }
                    commit(Array(frame.annotations.dropLast()))
                }
                .disabled(frame.annotations.isEmpty)
                Button(language.text("Frame leeren", "Clear frame"), role: .destructive) {
                    commit([])
                }
                .disabled(frame.annotations.isEmpty)
            }
        }
    }

    private func commit(_ annotations: [BoxAnnotation]) {
        guard annotations != frame.annotations else { return }
        undoStack.append(frame.annotations)
        if undoStack.count > 10 { undoStack.removeFirst(undoStack.count - 10) }
        redoStack.removeAll()
        onChange(annotations)
    }

    private func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(frame.annotations)
        if redoStack.count > 10 { redoStack.removeFirst(redoStack.count - 10) }
        onChange(previous)
    }

    private func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(frame.annotations)
        if undoStack.count > 10 { undoStack.removeFirst(undoStack.count - 10) }
        onChange(next)
    }

    private func zoomedRect(baseRect: CGRect, container: CGRect, scale: Double, offset: CGSize) -> CGRect {
        let size = CGSize(width: baseRect.width * scale, height: baseRect.height * scale)
        return CGRect(
            x: container.midX - size.width / 2 + offset.width,
            y: container.midY - size.height / 2 + offset.height,
            width: size.width,
            height: size.height
        )
    }

    private func clampedOffset(_ proposed: CGSize, baseRect: CGRect, container: CGRect, scale: Double) -> CGSize {
        let maxX = max(0, (baseRect.width * scale - container.width) / 2)
        let maxY = max(0, (baseRect.height * scale - container.height) / 2)
        return CGSize(
            width: min(max(proposed.width, -maxX), maxX),
            height: min(max(proposed.height, -maxY), maxY)
        )
    }

    private func aspectFit(imageSize: CGSize, in container: CGRect) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return container }
        let scale = min(container.width / imageSize.width, container.height / imageSize.height)
        let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(
            x: container.midX - size.width / 2,
            y: container.midY - size.height / 2,
            width: size.width,
            height: size.height
        )
    }

    private func screenRect(for annotation: BoxAnnotation, imageRect: CGRect) -> CGRect {
        let scaleX = imageRect.width / Double(frame.width)
        let scaleY = imageRect.height / Double(frame.height)
        return CGRect(
            x: imageRect.minX + annotation.x * scaleX,
            y: imageRect.minY + annotation.y * scaleY,
            width: annotation.width * scaleX,
            height: annotation.height * scaleY
        )
    }

    private func normalizedRect(from first: CGPoint, to second: CGPoint) -> CGRect {
        CGRect(
            x: min(first.x, second.x), y: min(first.y, second.y),
            width: abs(first.x - second.x), height: abs(first.y - second.y)
        )
    }

    private func clamp(_ point: CGPoint, to rect: CGRect) -> CGPoint {
        CGPoint(x: min(max(point.x, rect.minX), rect.maxX), y: min(max(point.y, rect.minY), rect.maxY))
    }

    private func color(for category: String) -> Color {
        switch category {
        case "ball", "puck": .yellow
        case "player": .cyan
        case "goalkeeper": .green
        case "referee": .orange
        default: .pink
        }
    }
}
