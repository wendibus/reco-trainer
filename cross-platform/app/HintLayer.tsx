import { useEffect, useRef, useState } from 'react';

const SHOW_DELAY_MS = 350;
const GAP = 8;
const EDGE = 8;

type Shown = { text: string; left: number; top: number };

function contains(rect: DOMRect, x: number, y: number) {
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

// Disabled buttons take no pointer events (globals.css) so all browsers behave alike; they are found by position.
function hintTargetAt(event: PointerEvent): Element | null {
  const disabled = [...document.querySelectorAll('button:disabled[data-hint]')].find((element) => contains(element.getBoundingClientRect(), event.clientX, event.clientY));
  return disabled ?? (event.target instanceof Element ? event.target.closest('[data-hint]') : null);
}

/** Shows the data-hint text of the button under the mouse or keyboard focus. */
export function HintLayer({ enabled }: { enabled: boolean }) {
  const [shown, setShown] = useState<Shown | null>(null);
  const [anchor, setAnchor] = useState<Element | null>(null);
  const bubble = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!enabled) return;
    let current: Element | null = null;
    let timer = 0;
    const aim = (next: Element | null) => {
      if (next === current) return;
      current = next;
      window.clearTimeout(timer);
      setAnchor(null);
      if (next) timer = window.setTimeout(() => setAnchor(next), SHOW_DELAY_MS);
    };
    const onPointerMove = (event: PointerEvent) => { if (event.pointerType !== 'touch') aim(hintTargetAt(event)); };
    // A click usually changes what the button does, so the hint stays away until the mouse leaves it.
    const onDismiss = () => { window.clearTimeout(timer); setAnchor(null); };
    const onFocusIn = (event: FocusEvent) => {
      if (event.target instanceof Element && event.target.matches(':focus-visible')) aim(event.target.closest('[data-hint]'));
    };
    const onFocusOut = () => aim(null);
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onDismiss(); };
    document.addEventListener('pointermove', onPointerMove, { passive: true });
    document.addEventListener('pointerdown', onDismiss, true);
    document.addEventListener('scroll', onDismiss, true);
    document.addEventListener('focusin', onFocusIn);
    document.addEventListener('focusout', onFocusOut);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      window.clearTimeout(timer);
      setAnchor(null);
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerdown', onDismiss, true);
      document.removeEventListener('scroll', onDismiss, true);
      document.removeEventListener('focusin', onFocusIn);
      document.removeEventListener('focusout', onFocusOut);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [enabled]);

  useEffect(() => {
    const text = anchor?.getAttribute('data-hint');
    setShown(anchor && text ? { text, left: -9999, top: -9999 } : null);
  }, [anchor]);

  // Placed after rendering, when the size of the bubble is known: above the button, below if there is no room.
  useEffect(() => {
    if (!anchor || !bubble.current || shown?.left !== -9999) return;
    const target = anchor.getBoundingClientRect();
    const own = bubble.current.getBoundingClientRect();
    const left = Math.min(Math.max(EDGE, target.left + target.width / 2 - own.width / 2), window.innerWidth - own.width - EDGE);
    const above = target.top - own.height - GAP;
    setShown({ ...shown, left, top: above >= EDGE ? above : target.bottom + GAP });
  }, [anchor, shown]);

  if (!enabled || !shown) return null;
  return <div ref={bubble} className="hint-bubble" role="tooltip" style={{ left: shown.left, top: shown.top }}>{shown.text}</div>;
}
