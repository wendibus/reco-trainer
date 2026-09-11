import AppKit
import SwiftUI

/// Lets the user type the field's real width/length and click its four corners
/// on a reference frame, so auto-label can later tell whether a detected
/// person's feet fall on the field (see ml_worker.py's field_membership_checker).
struct FieldGeometryEditor: View {
    let imageURL: URL
    let frameWidth: Int
    let frameHeight: Int
    let language: AppLanguage
    let existing: FieldGeometry?
    let onSave: (FieldGeometry) -> Void
    let onCancel: () -> Void

    @State private var realWidthText: String
    @State private var realLengthText: String
    /// Corners collected so far, in image pixel space (0...frameWidth/frameHeight).
    @State private var corners: [CGPoint]

    init(
        imageURL: URL,
        frameWidth: Int,
        frameHeight: Int,
        language: AppLanguage,
        existing: FieldGeometry?,
        onSave: @escaping (FieldGeometry) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.imageURL = imageURL
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.language = language
        self.existing = existing
        self.onSave = onSave
        self.onCancel = onCancel
        _realWidthText = State(initialValue: existing.map { formatMeters($0.realWidth) } ?? "")
        _realLengthText = State(initialValue: existing.map { formatMeters($0.realLength) } ?? "")
        _corners = State(initialValue: (existing?.corners ?? []).map {
            CGPoint(x: $0.x * Double(frameWidth), y: $0.y * Double(frameHeight))
        })
    }

    private static let cornerLabels: [(String, String, String, String)] = [
        ("Oben links", "Top left", "Superior izquierda", "En haut à gauche"),
        ("Oben rechts", "Top right", "Superior derecha", "En haut à droite"),
        ("Unten rechts", "Bottom right", "Inferior derecha", "En bas à droite"),
        ("Unten links", "Bottom left", "Inferior izquierda", "En bas à gauche"),
    ]

    private var realWidth: Double? { Double(realWidthText.replacingOccurrences(of: ",", with: ".")) }
    private var realLength: Double? { Double(realLengthText.replacingOccurrences(of: ",", with: ".")) }
    private var canSave: Bool {
        corners.count == 4 && (realWidth ?? 0) > 0 && (realLength ?? 0) > 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(language.text(
                "Spielfeld festlegen", "Set field boundaries",
                "Definir el campo", "Définir le terrain"
            )).font(.title2.bold())

            Text(language.text(
                "Damit „Automatisch markieren“ nur Personen berücksichtigt, die mit den Füßen auf dem Spielfeld stehen. Gib die echten Feldmaße ein und klicke dann die vier Eckpunkte im Bild an - in der Reihenfolge oben links, oben rechts, unten rechts, unten links.",
                "So \"Auto-label\" only considers people whose feet are standing on the field. Enter the field's real dimensions, then click its four corners in the image - in order: top left, top right, bottom right, bottom left.",
                "Para que «Marcado automático» solo considere a personas con los pies sobre el campo. Introduce las medidas reales del campo y luego haz clic en sus cuatro esquinas en la imagen, en este orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda.",
                "Pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain. Saisissez les dimensions réelles du terrain, puis cliquez sur ses quatre coins dans l’image - dans l’ordre : en haut à gauche, en haut à droite, en bas à droite, en bas à gauche."
            )).font(.callout).foregroundStyle(.secondary)

            HStack(spacing: 16) {
                LabeledContent(language.text("Breite (m)", "Width (m)", "Ancho (m)", "Largeur (m)")) {
                    TextField("", text: $realWidthText).frame(width: 80).textFieldStyle(.roundedBorder)
                }
                LabeledContent(language.text("Länge (m)", "Length (m)", "Largo (m)", "Longueur (m)")) {
                    TextField("", text: $realLengthText).frame(width: 80).textFieldStyle(.roundedBorder)
                }
                Spacer()
                if corners.count < 4 {
                    let label = Self.cornerLabels[corners.count]
                    Label(
                        language.text(
                            "Ecke \(corners.count + 1)/4: \(label.0)",
                            "Corner \(corners.count + 1)/4: \(label.1)",
                            "Esquina \(corners.count + 1)/4: \(label.2)",
                            "Coin \(corners.count + 1)/4 : \(label.3)"
                        ),
                        systemImage: "hand.tap"
                    ).foregroundStyle(.blue)
                } else {
                    Label(language.text("Alle 4 Ecken gesetzt", "All 4 corners set", "Las 4 esquinas listas", "Les 4 coins sont placés"), systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                Button(language.text("Zurücksetzen", "Reset", "Reiniciar", "Réinitialiser")) {
                    corners.removeAll()
                }
                .disabled(corners.isEmpty)
            }

            GeometryReader { geometry in
                let container = CGRect(origin: .zero, size: geometry.size)
                let imageSize = CGSize(width: frameWidth, height: frameHeight)
                let fitted = aspectFit(imageSize: imageSize, in: container)

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
                        let screenPoints = corners.map { screenPoint(for: $0, in: fitted) }
                        if screenPoints.count >= 2 {
                            var path = Path()
                            path.move(to: screenPoints[0])
                            for point in screenPoints.dropFirst() { path.addLine(to: point) }
                            if screenPoints.count == 4 { path.closeSubpath() }
                            context.stroke(path, with: .color(.yellow), lineWidth: 2)
                            if screenPoints.count == 4 {
                                context.fill(path, with: .color(.yellow.opacity(0.15)))
                            }
                        }
                        for (index, point) in screenPoints.enumerated() {
                            let dot = CGRect(x: point.x - 6, y: point.y - 6, width: 12, height: 12)
                            context.fill(Path(ellipseIn: dot), with: .color(.yellow))
                            context.stroke(Path(ellipseIn: dot), with: .color(.black), lineWidth: 1)
                            context.draw(Text("\(index + 1)").font(.caption.bold()).foregroundStyle(.black), at: CGPoint(x: point.x, y: point.y - 16))
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { location in
                        guard corners.count < 4, fitted.contains(location) else { return }
                        corners.append(clampedImagePoint(from: location, in: fitted))
                    }
                }
            }
            .frame(minHeight: 360)
            .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Spacer()
                Button(language.text("Abbrechen", "Cancel", "Cancelar", "Annuler"), action: onCancel)
                Button(language.text("Speichern", "Save", "Guardar", "Enregistrer")) {
                    guard let realWidth, let realLength else { return }
                    let geometry = FieldGeometry(
                        corners: corners.map {
                            FieldCorner(x: $0.x / Double(frameWidth), y: $0.y / Double(frameHeight))
                        },
                        realWidth: realWidth,
                        realLength: realLength
                    )
                    onSave(geometry)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave)
            }
        }
        .padding(24)
        .frame(width: 720, height: 640)
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

    private func screenPoint(for imagePoint: CGPoint, in fitted: CGRect) -> CGPoint {
        let scaleX = fitted.width / Double(frameWidth)
        let scaleY = fitted.height / Double(frameHeight)
        return CGPoint(x: fitted.minX + imagePoint.x * scaleX, y: fitted.minY + imagePoint.y * scaleY)
    }

    private func clampedImagePoint(from screenPoint: CGPoint, in fitted: CGRect) -> CGPoint {
        let scaleX = Double(frameWidth) / fitted.width
        let scaleY = Double(frameHeight) / fitted.height
        let x = min(max(0, Double(screenPoint.x - fitted.minX) * scaleX), Double(frameWidth))
        let y = min(max(0, Double(screenPoint.y - fitted.minY) * scaleY), Double(frameHeight))
        return CGPoint(x: x, y: y)
    }
}

private func formatMeters(_ value: Double) -> String {
    value.truncatingRemainder(dividingBy: 1) == 0 ? String(Int(value)) : String(value)
}
