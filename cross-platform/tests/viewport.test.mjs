// node --test tests/viewport.test.mjs  (Node 23.6+ runs the TypeScript module directly)
import test from 'node:test';
import assert from 'node:assert/strict';
import { MAX_ZOOM, clampZoom, placement, screenPixels, toImage, toScreen, viewBox } from '../app/viewport.ts';

const viewport = { width: 1058, height: 537 };
const image = { width: 1920, height: 1080 };
const close = (actual, expected, label) => assert.ok(Math.abs(actual - expected) < 1e-6, `${label}: ${actual} != ${expected}`);

test('zoom 1 letterboxes the whole frame', () => {
  const place = placement(viewport, image, { zoom: 1, pan: { x: 0, y: 0 } });
  close(place.scale, 537 / 1080, 'scale');
  close(place.y, 0, 'top');
  const [x, y, width, height] = viewBox(viewport, image, { zoom: 1, pan: { x: 0, y: 0 } }).split(' ').map(Number);
  assert.ok(x < 0 && Math.abs(y) < 1e-3 && Math.abs(height - 1080) < 1e-3 && width > 1920);
});

test('a box drawn over the same image region gets the same coordinates at every zoom', () => {
  const box = { x: 903.25, y: 501.5, width: 11.75, height: 12.25 };
  for (const zoom of [1, 2.5, 8, MAX_ZOOM]) {
    const view = { zoom, pan: { x: -37 * zoom, y: 12 * zoom } };
    // what the user does on screen: press at the top-left corner, release at the bottom-right corner
    const press = toScreen({ x: box.x, y: box.y }, viewport, image, view);
    const release = toScreen({ x: box.x + box.width, y: box.y + box.height }, viewport, image, view);
    const start = toImage(press, viewport, image, view), end = toImage(release, viewport, image, view);
    close(start.x, box.x, `x @${zoom}`); close(start.y, box.y, `y @${zoom}`);
    close(end.x - start.x, box.width, `width @${zoom}`); close(end.y - start.y, box.height, `height @${zoom}`);
  }
});

test('line width and label size stay constant on screen', () => {
  for (const zoom of [1, 3, MAX_ZOOM]) {
    const view = { zoom, pan: { x: 0, y: 0 } };
    const scale = placement(viewport, image, view).scale;
    close(screenPixels(1.5, viewport, image, view) * scale, 1.5, `1.5 px @${zoom}`);
  }
});

test('zoom is clamped to the allowed range', () => {
  assert.equal(clampZoom(0.2), 1);
  assert.equal(clampZoom(1e6), MAX_ZOOM);
  assert.equal(clampZoom(Number.NaN), 1);
});
