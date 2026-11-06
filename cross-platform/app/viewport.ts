// View math for the frame editor. The frame is drawn inside one SVG whose viewBox is the visible
// part of the image, so boxes are always stored and drawn in image pixels: zooming or panning
// only changes the viewBox, never a box. Kept free of React so tests can run it with plain Node.

export type Size = { width: number; height: number };
export type Point = { x: number; y: number };
export type View = { zoom: number; pan: Point };

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 20;
/** One +/- click: three clicks double or halve the zoom, so every step feels the same (1 → 1.26 → 1.59 → 2 … 20). */
export const ZOOM_STEP = 2 ** (1 / 3);

/** Next zoom for `steps` button clicks (negative = out); lands exactly on the limits. */
export function steppedZoom(zoom: number, steps: number): number {
  const next = zoom * ZOOM_STEP ** steps;
  return clampZoom(Math.abs(next - Math.round(next)) < 0.02 ? Math.round(next) : next);
}

/** "1.26×", "4.0×", "20×" */
export function zoomLabel(zoom: number): string {
  return `${zoom < 2 ? zoom.toFixed(2) : zoom < 10 ? zoom.toFixed(1) : Math.round(zoom)}×`;
}

export const clampZoom = (zoom: number) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number.isFinite(zoom) ? zoom : MIN_ZOOM));

/** Scale at zoom 1: the whole frame fits into the viewport (letterboxed). */
export function fitScale(viewport: Size, image: Size): number {
  if (viewport.width <= 0 || viewport.height <= 0 || image.width <= 0 || image.height <= 0) return 1;
  return Math.min(viewport.width / image.width, viewport.height / image.height);
}

/** Screen pixels per image pixel and where the image's top-left corner sits in the viewport.
 *  pan is the offset of the image centre from the viewport centre, in screen pixels. */
export function placement(viewport: Size, image: Size, view: View): { scale: number; x: number; y: number } {
  const scale = fitScale(viewport, image) * view.zoom;
  return {
    scale,
    x: viewport.width / 2 + view.pan.x - (image.width * scale) / 2,
    y: viewport.height / 2 + view.pan.y - (image.height * scale) / 2,
  };
}

/** The SVG viewBox (in image pixels) that shows exactly what the viewport sees. */
export function viewBox(viewport: Size, image: Size, view: View): string {
  const { scale, x, y } = placement(viewport, image, view);
  return [-x / scale, -y / scale, viewport.width / scale, viewport.height / scale].map((value) => +value.toFixed(4)).join(' ');
}

/** Viewport-relative screen point -> image pixel. */
export function toImage(point: Point, viewport: Size, image: Size, view: View): Point {
  const { scale, x, y } = placement(viewport, image, view);
  return { x: (point.x - x) / scale, y: (point.y - y) / scale };
}

/** Image pixel -> viewport-relative screen point. */
export function toScreen(point: Point, viewport: Size, image: Size, view: View): Point {
  const { scale, x, y } = placement(viewport, image, view);
  return { x: x + point.x * scale, y: y + point.y * scale };
}

/** Screen-constant length (stroke, handle, label size) expressed in image pixels. */
export function screenPixels(pixels: number, viewport: Size, image: Size, view: View): number {
  return pixels / placement(viewport, image, view).scale;
}

/** Zoom so that the image point under `point` (viewport-relative, e.g. the mouse) stays put.
 *  Back at zoom 1 the frame is centred again. */
export function zoomAt(point: Point, nextZoom: number, viewport: Size, image: Size, view: View): View {
  const zoom = clampZoom(nextZoom);
  if (zoom === MIN_ZOOM) return { zoom, pan: { x: 0, y: 0 } };
  const anchor = toImage(point, viewport, image, view);
  const scale = fitScale(viewport, image) * zoom;
  return {
    zoom,
    pan: {
      x: point.x - anchor.x * scale - viewport.width / 2 + (image.width * scale) / 2,
      y: point.y - anchor.y * scale - viewport.height / 2 + (image.height * scale) / 2,
    },
  };
}

/** Wheel or trackpad delta -> zoom factor. Lines and pages are converted to pixels so a mouse
 *  notch and a trackpad swipe feel alike. */
export function wheelFactor(deltaY: number, deltaMode = 0): number {
  const pixels = deltaY * (deltaMode === 1 ? 16 : deltaMode === 2 ? 400 : 1);
  return Math.exp(-Math.max(-300, Math.min(300, pixels)) * 0.0022);
}
