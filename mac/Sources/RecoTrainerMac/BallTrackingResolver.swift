import Foundation

/// How a given frame's ball marker should be drawn, named after reco cam's
/// own tracker vocabulary (Coaster's Tracking/Coasting/Lost) since this
/// mirrors its methodology: nearest-known-position tracking with a bounded
/// hold, extended here to also bridge a gap by blending toward a real future
/// detection when the whole clip is already available (see resolve(...)).
enum BallTrackState: String, Sendable {
    /// A real detection landed on this frame.
    case detected
    /// No detection this frame, but a real detection exists both before and
    /// after within the lookahead window - the shown position blends between
    /// them, weighted by how close this frame is to each.
    case interpolated
    /// No detection this frame and no future one within the window either,
    /// but a past detection exists within the window - the shown position
    /// holds at that last known spot (reco cam's "Coasting").
    case coasting
    /// No detection within the window in either direction (including before
    /// the very first real detection, which reco cam also shows nothing
    /// for) - nothing is drawn.
    case lost
}

struct ResolvedBallPosition: Sendable, Equatable {
    let state: BallTrackState
    let x: Double?
    let y: Double?
}

enum BallTrackingResolver {
    /// Resolves raw per-frame ball detections (nil where the model found
    /// nothing) into a per-frame track. Pure and index-based, no I/O - the
    /// expensive part (running the model) already happened once server-side;
    /// this just lets the UI react instantly when the "how far to look"
    /// slider changes, without re-running inference.
    static func resolve(detections: [(x: Double, y: Double)?], lookaheadFrames: Int) -> [ResolvedBallPosition] {
        let window = max(0, lookaheadFrames)
        var result: [ResolvedBallPosition] = []
        result.reserveCapacity(detections.count)
        for index in detections.indices {
            if let point = detections[index] {
                result.append(ResolvedBallPosition(state: .detected, x: point.x, y: point.y))
                continue
            }

            var last: (x: Double, y: Double, distance: Int)?
            var back = index - 1
            while back >= 0, index - back <= window {
                if let point = detections[back] {
                    last = (point.x, point.y, index - back)
                    break
                }
                back -= 1
            }

            guard let last else {
                result.append(ResolvedBallPosition(state: .lost, x: nil, y: nil))
                continue
            }

            var next: (x: Double, y: Double, distance: Int)?
            var forward = index + 1
            while forward < detections.count, forward - index <= window {
                if let point = detections[forward] {
                    next = (point.x, point.y, forward - index)
                    break
                }
                forward += 1
            }

            guard let next else {
                result.append(ResolvedBallPosition(state: .coasting, x: last.x, y: last.y))
                continue
            }

            let span = Double(last.distance + next.distance)
            let t = span > 0 ? Double(last.distance) / span : 0
            let x = last.x + (next.x - last.x) * t
            let y = last.y + (next.y - last.y) * t
            result.append(ResolvedBallPosition(state: .interpolated, x: x, y: y))
        }
        return result
    }
}
