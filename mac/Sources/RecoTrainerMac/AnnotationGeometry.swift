import CoreGraphics

/// Picks the smallest annotation whose box contains the given point (in the frame's
/// own image-space coordinates, not screen coordinates). Preferring the smallest match
/// keeps a small box nested inside a larger one - e.g. a ball box inside a player box -
/// individually selectable by clicking directly on it, instead of always hitting the
/// larger box underneath.
func annotation(at point: CGPoint, among annotations: [BoxAnnotation]) -> BoxAnnotation? {
    let candidates = annotations.filter { contains($0, point) }
    return candidates.min { $0.width * $0.height < $1.width * $1.height }
}

private func contains(_ annotation: BoxAnnotation, _ point: CGPoint) -> Bool {
    let x = Double(point.x)
    let y = Double(point.y)
    let withinX = x >= annotation.x && x <= annotation.x + annotation.width
    let withinY = y >= annotation.y && y <= annotation.y + annotation.height
    return withinX && withinY
}

/// Moves an annotation by an image-space delta, clamped so it stays fully inside the
/// frame. Used when committing a drag-to-move gesture.
func clampedMove(
    of annotation: BoxAnnotation,
    byImageDelta delta: CGSize,
    imageWidth: Int,
    imageHeight: Int
) -> BoxAnnotation {
    var moved = annotation
    let maxX = max(0.0, Double(imageWidth) - annotation.width)
    let maxY = max(0.0, Double(imageHeight) - annotation.height)
    moved.x = min(max(0.0, annotation.x + Double(delta.width)), maxX)
    moved.y = min(max(0.0, annotation.y + Double(delta.height)), maxY)
    return moved
}
