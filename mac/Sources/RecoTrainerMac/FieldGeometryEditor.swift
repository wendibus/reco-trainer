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
    @State private var dragIndex: Int?
    @State private var dragActive = false

    private static let maxPoints = 32

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
    /// Exactly four points are the corners of a rectangular field and need its real
    /// size; more points trace any other outline, where the size is optional.
    private var canSave: Bool {
        let widthEntered = !realWidthText.trimmingCharacters(in: .whitespaces).isEmpty
        let lengthEntered = !realLengthText.trimmingCharacters(in: .whitespaces).isEmpty
        if corners.count == 4 { return (realWidth ?? 0) > 0 && (realLength ?? 0) > 0 }
        guard corners.count > 4 else { return false }
        return (!widthEntered || (realWidth ?? -1) >= 0) && (!lengthEntered || (realLength ?? -1) >= 0)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(language.text(
                "Spielfeld festlegen", "Set field boundaries",
                "Definir el campo", "Définir le terrain"
            )).font(.title2.bold())

            Text(language.text(
                "Damit „Automatisch markieren“ nur Personen berücksichtigt, die mit den Füßen auf dem Spielfeld stehen. Klicke die Eckpunkte des Spielfelds im Bild an - bei einem rechteckigen Feld die vier Ecken in dieser Reihenfolge: oben links, oben rechts, unten rechts, unten links - und gib die echten Feldmaße ein. Bei anderen Formen setzt du einfach weitere Punkte rund um den Rand (bis zu 32); die Maße sind dann optional. Punkte lassen sich verschieben, der letzte Punkt kann wieder entfernt werden.",
                "So \"Auto-label\" only considers people whose feet are standing on the field. Click the field's corner points in the image - for a rectangular field its four corners in this order: top left, top right, bottom right, bottom left - and enter the field's real dimensions. For other shapes just keep adding points around the outline (up to 32); the dimensions are then optional. Drag a point to move it, or remove the last point again.",
                "Para que «Marcado automático» solo considere a personas con los pies sobre el campo. Haz clic en los puntos del contorno del campo en la imagen: en un campo rectangular, sus cuatro esquinas en este orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda, e introduce las medidas reales. Para otras formas sigue añadiendo puntos alrededor del borde (hasta 32); las medidas son entonces opcionales. Arrastra un punto para moverlo o quita el último.",
                "Pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain. Cliquez sur les points du contour du terrain dans l’image - pour un terrain rectangulaire, ses quatre coins dans l’ordre : en haut à gauche, en haut à droite, en bas à droite, en bas à gauche - et saisissez les dimensions réelles. Pour d’autres formes, continuez à ajouter des points le long du bord (jusqu’à 32) ; les dimensions sont alors facultatives. Faites glisser un point pour le déplacer ou supprimez le dernier."
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
                } else if corners.count == 4 {
                    Label(language.text("Alle 4 Ecken gesetzt", "All 4 corners set", "Las 4 esquinas listas", "Les 4 coins sont placés"), systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else {
                    Label(language.text("Punkte: \(corners.count)", "Points: \(corners.count)", "Puntos: \(corners.count)", "Points : \(corners.count)"), systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                Button(language.text("Letzten Punkt entfernen", "Remove last point", "Quitar el último punto", "Supprimer le dernier point")) {
                    _ = corners.popLast()
                }
                .disabled(corners.isEmpty)
                Button(language.text("Zurücksetzen", "Reset", "Reiniciar", "Réinitialiser")) {
                    corners.removeAll()
                }
                .disabled(corners.isEmpty)
            }

            Text(language.text(
                "Klicken, um weitere Punkte zu setzen; Punkt ziehen zum Verschieben",
                "Click to add more points, drag a point to move it",
                "Haz clic para añadir más puntos; arrastra un punto para moverlo",
                "Cliquez pour ajouter des points, faites glisser un point pour le déplacer"
            )).font(.caption).foregroundStyle(.secondary)

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
                            language.text("Frame fehlt", "Frame is missing", "Falta el fotograma", "Image manquante"),
                            systemImage: "photo.badge.exclamationmark"
                        )
                    }
                    Canvas { context, _ in
                        let screenPoints = corners.map { screenPoint(for: $0, in: fitted) }
                        if screenPoints.count >= 2 {
                            var path = Path()
                            path.move(to: screenPoints[0])
                            for point in screenPoints.dropFirst() { path.addLine(to: point) }
                            if screenPoints.count >= 4 { path.closeSubpath() }
                            context.stroke(path, with: .color(.yellow), lineWidth: 2)
                            if screenPoints.count >= 4 {
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
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                if !dragActive {
                                    dragActive = true
                                    dragIndex = nearestCornerIndex(to: value.startLocation, in: fitted)
                                }
                                if let index = dragIndex, corners.indices.contains(index) {
                                    corners[index] = clampedImagePoint(from: value.location, in: fitted)
                                }
                            }
                            .onEnded { value in
                                defer { dragActive = false; dragIndex = nil }
                                let travelled = hypot(value.location.x - value.startLocation.x, value.location.y - value.startLocation.y)
                                guard dragIndex == nil, travelled < 5, corners.count < Self.maxPoints, fitted.contains(value.startLocation) else { return }
                                corners.append(clampedImagePoint(from: value.startLocation, in: fitted))
                            }
                    )
                }
            }
            .frame(minHeight: 360)
            .background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Spacer()
                Button(language.text("Abbrechen", "Cancel", "Cancelar", "Annuler"), action: onCancel)
                Button(language.text("Speichern", "Save", "Guardar", "Enregistrer")) {
                    let width = realWidth ?? 0
                    let length = realLength ?? 0
                    let geometry = FieldGeometry(
                        corners: corners.map {
                            FieldCorner(x: $0.x / Double(frameWidth), y: $0.y / Double(frameHeight))
                        },
                        realWidth: width,
                        realLength: length
                    )
                    onSave(geometry)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave)
            }
        }
        .padding(24)
        .frame(width: 760, height: 720)
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

    /// The marked point under a press, if any (within a finger-sized radius), so a
    /// press on a point drags it instead of adding a new one.
    private func nearestCornerIndex(to location: CGPoint, in fitted: CGRect) -> Int? {
        let hits = corners.enumerated().compactMap { index, corner -> (Int, CGFloat)? in
            let point = screenPoint(for: corner, in: fitted)
            let distance = hypot(point.x - location.x, point.y - location.y)
            return distance <= 14 ? (index, distance) : nil
        }
        return hits.min { $0.1 < $1.1 }?.0
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
