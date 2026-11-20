type Language = 'de' | 'en' | 'es' | 'fr';

export type FrameSort = 'time' | 'fewest' | 'most';
export type ClassMode = 'any' | 'with' | 'without';
export type FrameRefine = { query: string; minBoxes: string; maxBoxes: string; category: string; classMode: ClassMode; sort: FrameSort; incomplete: boolean };
export const NO_REFINE: FrameRefine = { query: '', minBoxes: '', maxBoxes: '', category: '', classMode: 'any', sort: 'time', incomplete: false };

const REFINE_KEY = 'reco-frame-refine-v1';

/** The filters the person last used, so they survive a restart. Storage may be blocked; then the list simply starts unfiltered. */
export function loadRefine(): FrameRefine {
  try {
    const raw = JSON.parse(window.localStorage.getItem(REFINE_KEY) ?? 'null');
    if (raw && typeof raw === 'object') {
      return {
        query: typeof raw.query === 'string' ? raw.query : '',
        minBoxes: typeof raw.minBoxes === 'string' ? raw.minBoxes.replace(/[^0-9]/g, '') : '',
        maxBoxes: typeof raw.maxBoxes === 'string' ? raw.maxBoxes.replace(/[^0-9]/g, '') : '',
        category: typeof raw.category === 'string' ? raw.category : '',
        classMode: raw.classMode === 'with' || raw.classMode === 'without' ? raw.classMode : 'any',
        sort: raw.sort === 'fewest' || raw.sort === 'most' ? raw.sort : 'time',
        incomplete: raw.incomplete === true,
      };
    }
  } catch { /* storage unavailable or damaged */ }
  return NO_REFINE;
}

export function saveRefine(refine: FrameRefine): void {
  try { window.localStorage.setItem(REFINE_KEY, JSON.stringify(refine)); } catch { /* storage unavailable */ }
}

/** Classes that nearly every labelled image contains, such as players. An image that has boxes but lacks one of them is probably only half labelled. */
export function expectedClasses(frames: RefinableFrame[], classes: string[]): string[] {
  const labelled = frames.filter((frame) => frame.annotations.length && frame.reviewStatus !== 'candidate');
  const base = labelled.length >= 8 ? labelled : frames.filter((frame) => frame.annotations.length);
  if (base.length < 8) return [];
  return classes.filter((name) => base.filter((frame) => frame.annotations.some((box) => box.category === name)).length / base.length >= 0.6);
}

/** Images with at least one box that miss an expected class. */
export function incompleteFrameIds(frames: RefinableFrame[], expected: string[]): Set<string> {
  const ids = new Set<string>();
  if (!expected.length) return ids;
  for (const frame of frames) {
    if (!frame.annotations.length) continue;
    if (expected.some((name) => !frame.annotations.some((box) => box.category === name))) ids.add(frame.id);
  }
  return ids;
}

type RefinableFrame = { id: string; videoName: string; timestamp: number; reviewStatus?: string; annotations: Array<{ category: string }> };

const TEXT: Record<Language, { search: string; searchHint: string; boxes: string; from: string; to: string; boxesHint: string; cls: string; clsHint: string; any: string; with: (c: string) => string; without: (c: string) => string; sort: string; sortHint: string; sorts: Record<FrameSort, string>; reset: string; resetHint: string; shown: (n: number, total: number) => string; incomplete: (n: number) => string; incompleteHint: (names: string) => string; adopt: string; adoptHint: string }> = {
  de: { search: 'Video oder Zeit suchen', searchHint: 'Filtert die Liste nach Videoname oder Zeit, zum Beispiel „halbzeit“ oder „12:30“.', boxes: 'Boxen', from: 'von', to: 'bis', boxesHint: 'Zeigt nur Bilder mit einer Anzahl Boxen in diesem Bereich. 0 bis 0 findet Bilder ohne Boxen, 1 bis 3 Bilder mit wenigen.', cls: 'Klasse', clsHint: 'Findet Bilder, die eine Klasse enthalten oder ihr fehlen, zum Beispiel Bilder mit Ball, aber ohne Spieler.', any: 'egal', with: (c) => `mit ${c}`, without: (c) => `ohne ${c}`, sort: 'Sortierung', sortHint: 'Ändert die Reihenfolge der Bilderliste. Die Pfeiltasten folgen dieser Reihenfolge.', sorts: { time: 'Video und Zeit', fewest: 'Wenigste Boxen zuerst', most: 'Meiste Boxen zuerst' }, reset: 'Filter zurücksetzen', resetHint: 'Setzt Suche, Boxen, Klasse und Sortierung zurück.', shown: (n, total) => `${n} von ${total} Bildern`, incomplete: (n) => `Klasse fehlt (${n})`, incompleteHint: (names) => `Bilder mit Boxen, in denen eine übliche Klasse fehlt (${names}), die fast alle anderen markierten Bilder haben. Oft ist dort nur der Ball markiert, aber keine Spieler.`, adopt: 'Als Training übernehmen', adoptHint: 'Nimmt dieses Bild mit seinen Boxen ins Training. Die Boxen gelten danach als bestätigt.' },
  en: { search: 'Search video or time', searchHint: 'Filters the list by video name or time, for example “half” or “12:30”.', boxes: 'Boxes', from: 'from', to: 'to', boxesHint: 'Shows only images with a number of boxes in this range. 0 to 0 finds images without boxes, 1 to 3 images with only a few.', cls: 'Class', clsHint: 'Finds images that contain a class or lack it, for example images with a ball but without players.', any: 'any', with: (c) => `with ${c}`, without: (c) => `without ${c}`, sort: 'Sort', sortHint: 'Changes the order of the image list. The arrow keys follow this order.', sorts: { time: 'Video and time', fewest: 'Fewest boxes first', most: 'Most boxes first' }, reset: 'Reset filters', resetHint: 'Resets search, boxes, class and sorting.', shown: (n, total) => `${n} of ${total} images`, incomplete: (n) => `Class missing (${n})`, incompleteHint: (names) => `Images with boxes that lack a class (${names}) which almost all other labelled images have. Often only the ball is marked there, but no players.`, adopt: 'Use for training', adoptHint: 'Puts this image with its boxes into training. The boxes count as confirmed afterwards.' },
  es: { search: 'Buscar vídeo o tiempo', searchHint: 'Filtra la lista por nombre de vídeo o tiempo, por ejemplo «descanso» o «12:30».', boxes: 'Cuadros', from: 'de', to: 'a', boxesHint: 'Muestra solo imágenes con una cantidad de cuadros en este rango. 0 a 0 encuentra imágenes sin cuadros, 1 a 3 imágenes con pocos.', cls: 'Clase', clsHint: 'Encuentra imágenes que contienen una clase o no la tienen, por ejemplo imágenes con balón pero sin jugadores.', any: 'cualquiera', with: (c) => `con ${c}`, without: (c) => `sin ${c}`, sort: 'Orden', sortHint: 'Cambia el orden de la lista de imágenes. Las teclas de flecha siguen este orden.', sorts: { time: 'Vídeo y tiempo', fewest: 'Menos cuadros primero', most: 'Más cuadros primero' }, reset: 'Restablecer filtros', resetHint: 'Restablece búsqueda, cuadros, clase y orden.', shown: (n, total) => `${n} de ${total} imágenes`, incomplete: (n) => `Falta una clase (${n})`, incompleteHint: (names) => `Imágenes con cuadros a las que les falta una clase (${names}) que tienen casi todas las demás imágenes marcadas. A menudo solo está marcado el balón, pero ningún jugador.`, adopt: 'Usar para entrenar', adoptHint: 'Incluye esta imagen con sus cuadros en el entrenamiento. Después los cuadros cuentan como confirmados.' },
  fr: { search: 'Chercher vidéo ou temps', searchHint: 'Filtre la liste par nom de vidéo ou temps, par exemple « mi-temps » ou « 12:30 ».', boxes: 'Boîtes', from: 'de', to: 'à', boxesHint: 'N’affiche que les images dont le nombre de boîtes est dans cette plage. 0 à 0 trouve les images sans boîtes, 1 à 3 celles qui en ont peu.', cls: 'Classe', clsHint: 'Trouve les images qui contiennent une classe ou qui en manquent, par exemple des images avec ballon mais sans joueurs.', any: 'indifférent', with: (c) => `avec ${c}`, without: (c) => `sans ${c}`, sort: 'Tri', sortHint: 'Change l’ordre de la liste d’images. Les touches fléchées suivent cet ordre.', sorts: { time: 'Vidéo et temps', fewest: 'Moins de boîtes d’abord', most: 'Plus de boîtes d’abord' }, reset: 'Réinitialiser les filtres', resetHint: 'Réinitialise la recherche, les boîtes, la classe et le tri.', shown: (n, total) => `${n} images sur ${total}`, incomplete: (n) => `Classe manquante (${n})`, incompleteHint: (names) => `Images avec des boîtes auxquelles manque une classe (${names}) que presque toutes les autres images annotées ont. Souvent seul le ballon est marqué, sans joueurs.`, adopt: 'Utiliser pour l’entraînement', adoptHint: 'Met cette image avec ses boîtes dans l’entraînement. Les boîtes comptent ensuite comme confirmées.' },
};

export function adoptLabel(language: Language): { label: string; hint: string } {
  return { label: TEXT[language].adopt, hint: TEXT[language].adoptHint };
}

export function isRefined(refine: FrameRefine): boolean {
  return Boolean(refine.incomplete || refine.query.trim() || refine.minBoxes !== '' || refine.maxBoxes !== '' || refine.classMode !== 'any' || refine.sort !== 'time');
}

function clock(seconds: number): string {
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

/** Search, box-count, class and sort refinements for the image list. `keepId` stays listed while it is selected, so fixing an image does not make it vanish under the cursor. */
export function refineFrames<T extends RefinableFrame>(frames: T[], refine: FrameRefine, keepId?: string | null, incomplete?: Set<string>): T[] {
  const query = refine.query.trim().toLowerCase();
  const min = refine.minBoxes === '' ? null : Number(refine.minBoxes);
  const max = refine.maxBoxes === '' ? null : Number(refine.maxBoxes);
  const matches = (frame: T) => {
    const count = frame.annotations.length;
    if (refine.incomplete && !incomplete?.has(frame.id)) return false;
    if (query && !`${frame.videoName} ${clock(frame.timestamp)}`.toLowerCase().includes(query)) return false;
    if (min !== null && Number.isFinite(min) && count < min) return false;
    if (max !== null && Number.isFinite(max) && count > max) return false;
    if (refine.category && refine.classMode !== 'any') {
      const has = frame.annotations.some((box) => box.category === refine.category);
      if ((refine.classMode === 'with') !== has) return false;
    }
    return true;
  };
  const listed = frames.filter((frame) => frame.id === keepId || matches(frame));
  if (refine.sort === 'time') return listed;
  const direction = refine.sort === 'fewest' ? 1 : -1;
  return [...listed].sort((left, right) => (left.annotations.length - right.annotations.length) * direction);
}

/** Search, box-count range, class presence and sort order above the image list. */
export function FrameFilters({ language, classes, expected, incompleteCount, refine, shown, total, onChange }: { language: Language; classes: string[]; expected: string[]; incompleteCount: number; refine: FrameRefine; shown: number; total: number; onChange: (next: FrameRefine) => void }) {
  const t = TEXT[language];
  const set = (patch: Partial<FrameRefine>) => onChange({ ...refine, ...patch });
  const number = (value: string) => value.replace(/[^0-9]/g, '').slice(0, 3);
  return <div className="frame-filters">
    {incompleteCount > 0 && <button data-hint={t.incompleteHint(expected.join(', '))} type="button" className={`split-chip frame-incomplete${refine.incomplete ? ' active' : ''}`} aria-pressed={refine.incomplete} onClick={() => set({ incomplete: !refine.incomplete })}>{t.incomplete(incompleteCount)}</button>}
    <input data-hint={t.searchHint} type="search" value={refine.query} placeholder={t.search} aria-label={t.search} onChange={(event) => set({ query: event.target.value })} />
    <div className="frame-filter-row" data-hint={t.boxesHint}>
      <span>{t.boxes}</span>
      <input type="text" inputMode="numeric" value={refine.minBoxes} placeholder={t.from} aria-label={`${t.boxes} ${t.from}`} onChange={(event) => set({ minBoxes: number(event.target.value) })} />
      <input type="text" inputMode="numeric" value={refine.maxBoxes} placeholder={t.to} aria-label={`${t.boxes} ${t.to}`} onChange={(event) => set({ maxBoxes: number(event.target.value) })} />
    </div>
    {classes.length > 0 && <label className="frame-filter-row" data-hint={t.clsHint}>
      <span>{t.cls}</span>
      <select value={refine.classMode === 'any' ? '' : `${refine.classMode}:${refine.category}`} onChange={(event) => {
        const [mode, ...rest] = event.target.value.split(':');
        set(mode ? { classMode: mode as ClassMode, category: rest.join(':') } : { classMode: 'any', category: '' });
      }}>
        <option value="">{t.any}</option>
        {classes.map((name) => <option key={`with:${name}`} value={`with:${name}`}>{t.with(name)}</option>)}
        {classes.map((name) => <option key={`without:${name}`} value={`without:${name}`}>{t.without(name)}</option>)}
      </select>
    </label>}
    <label className="frame-filter-row" data-hint={t.sortHint}>
      <span>{t.sort}</span>
      <select value={refine.sort} onChange={(event) => set({ sort: event.target.value as FrameSort })}>
        {(Object.keys(t.sorts) as FrameSort[]).map((key) => <option key={key} value={key}>{t.sorts[key]}</option>)}
      </select>
    </label>
    {isRefined(refine) && <div className="frame-filter-foot">
      <small>{t.shown(shown, total)}</small>
      <button data-hint={t.resetHint} type="button" className="secondary-button" onClick={() => onChange(NO_REFINE)}>{t.reset}</button>
    </div>}
  </div>;
}

/** One small dot per class an image contains, in the class colour, so half-labelled images stand out in the list. */
export function ClassDots({ names, colorOf, present }: { names: string[]; colorOf: (name: string) => string; present: Array<{ category: string }> }) {
  const counts = new Map<string, number>();
  for (const box of present) counts.set(box.category, (counts.get(box.category) ?? 0) + 1);
  const shown = names.filter((name) => counts.has(name));
  if (!shown.length) return null;
  return <span className="class-dots" aria-label={shown.map((name) => `${name} ${counts.get(name)}`).join(', ')}>{shown.map((name) => <i key={name} data-hint={`${name}: ${counts.get(name)}`} style={{ background: colorOf(name) }} />)}</span>;
}
