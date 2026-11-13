type Language = 'de' | 'en' | 'es' | 'fr';

export type SplitRole = 'train' | 'valid' | 'test';
export type SplitPreview = { roles: Record<string, SplitRole>; counts: Record<SplitRole, number>; warning?: string | null; problems: string[]; newImages?: number };
export type SplitFilterValue = 'all' | SplitRole | 'none';

const TEXT: Record<Language, { all: string; none: string; roles: Record<SplitRole, string>; hint: string; noneHint: string; summary: (counts: Record<SplitRole, number>) => string; perVideo: string }> = {
  de: {
    all: 'Alle', none: 'Nicht im Training', roles: { train: 'Training', valid: 'Validierung', test: 'Test' },
    hint: 'Zeigt nur die Bilder dieser Rolle. Die Rollen ergeben sich aus den Videos: Ein Video landet immer ganz in einer Rolle, damit fast gleiche Nachbarbilder nicht Training und Bewertung vermischen.',
    noneHint: 'Bilder, die das nächste Training nicht nutzt: noch ohne Markierung und nicht als leer bestätigt, Prüfkandidaten und der unabhängige Test.',
    summary: (c) => `Nächstes Training: ${c.train} Training · ${c.valid} Validierung · ${c.test} Test`,
    perVideo: 'Aufgeteilt nach ganzen Videos: ab drei Videos wird je eins Validierung und Test.',
  },
  en: {
    all: 'All', none: 'Not in training', roles: { train: 'Training', valid: 'Validation', test: 'Test' },
    hint: 'Shows only the images in this role. Roles follow the videos: a video always goes into one role as a whole, so near-identical neighboring frames never mix training and evaluation.',
    noneHint: 'Images the next training does not use: not labelled yet and not confirmed as empty, review candidates and the independent test.',
    summary: (c) => `Next training: ${c.train} training · ${c.valid} validation · ${c.test} test`,
    perVideo: 'Split by whole videos: from three videos on, one becomes validation and one test.',
  },
  es: {
    all: 'Todas', none: 'Fuera del entrenamiento', roles: { train: 'Entrenamiento', valid: 'Validación', test: 'Prueba' },
    hint: 'Muestra solo las imágenes de este rol. Los roles siguen a los vídeos: un vídeo siempre va entero a un rol, para que fotogramas vecinos casi iguales no mezclen entrenamiento y evaluación.',
    noneHint: 'Imágenes que el próximo entrenamiento no usa: aún sin marcar y no confirmadas como vacías, candidatos de revisión y la prueba independiente.',
    summary: (c) => `Próximo entrenamiento: ${c.train} entrenamiento · ${c.valid} validación · ${c.test} prueba`,
    perVideo: 'División por vídeos completos: a partir de tres vídeos, uno es validación y otro prueba.',
  },
  fr: {
    all: 'Toutes', none: 'Hors entraînement', roles: { train: 'Entraînement', valid: 'Validation', test: 'Test' },
    hint: 'N’affiche que les images de ce rôle. Les rôles suivent les vidéos : une vidéo va toujours entière dans un rôle, pour que des images voisines presque identiques ne mélangent pas entraînement et évaluation.',
    noneHint: 'Images que le prochain entraînement n’utilise pas : pas encore annotées ni confirmées vides, candidats de révision et le test indépendant.',
    summary: (c) => `Prochain entraînement : ${c.train} entraînement · ${c.valid} validation · ${c.test} test`,
    perVideo: 'Répartition par vidéos entières : à partir de trois vidéos, une sert à la validation et une au test.',
  },
};

const ROLE_TEXT: Record<Language, { title: string; intro: string; auto: string; now: string; hint: string; images: string }> = {
  de: { title: 'Rollen der Videos', intro: 'Ein Video liegt immer ganz in einer Rolle. Wähle, welches Video trainiert, validiert oder testet. Fehlt Validierung oder Test, sucht die App sie aus den übrigen Videos aus; ein Video bleibt zum Trainieren.', auto: 'Automatisch', now: 'jetzt', hint: 'Legt fest, welche Rolle dieses Video im nächsten Training hat. „Automatisch“ überlässt es der App.', images: 'Bilder' },
  en: { title: 'Roles of the videos', intro: 'A video always goes into one role as a whole. Choose which video trains, validates or tests. If validation or test is missing, the app picks it from the remaining videos; one video is left for training.', auto: 'Automatic', now: 'now', hint: 'Sets the role this video has in the next training. “Automatic” leaves it to the app.', images: 'images' },
  es: { title: 'Roles de los vídeos', intro: 'Un vídeo siempre va entero a un rol. Elige qué vídeo entrena, valida o prueba. Si falta validación o prueba, la app la elige entre los demás vídeos; uno queda para entrenar.', auto: 'Automático', now: 'ahora', hint: 'Fija el rol de este vídeo en el próximo entrenamiento. «Automático» lo deja a la app.', images: 'imágenes' },
  fr: { title: 'Rôles des vidéos', intro: 'Une vidéo va toujours entière dans un rôle. Choisissez quelle vidéo entraîne, valide ou teste. S’il manque la validation ou le test, l’app les choisit parmi les autres vidéos ; une vidéo reste pour l’entraînement.', auto: 'Automatique', now: 'actuellement', hint: 'Définit le rôle de cette vidéo au prochain entraînement. « Automatique » laisse l’app décider.', images: 'images' },
};

type RoleFrame = { id: string; videoID?: string; videoName?: string; heldOut?: boolean };

/** The videos of the project with the role each has in the next training, and a choice per video. */
export function VideoRoles({ language, frames, splits, videoRoles, busy, onChange }: { language: Language; frames: RoleFrame[]; splits: SplitPreview | null | undefined; videoRoles: Record<string, string> | undefined; busy: boolean; onChange: (videoID: string, role: SplitRole | 'auto') => void }) {
  const t = ROLE_TEXT[language];
  const names = TEXT[language].roles;
  const videos = new Map<string, { name: string; ids: string[] }>();
  for (const frame of frames) {
    if (!frame.videoID || frame.heldOut || frame.videoID.startsWith('import-')) continue;
    const entry = videos.get(frame.videoID) ?? { name: frame.videoName || frame.videoID, ids: [] };
    entry.ids.push(frame.id);
    videos.set(frame.videoID, entry);
  }
  if (!splits || videos.size < 2) return null;
  const current = (ids: string[]): SplitRole | null => {
    const counts: Record<string, number> = {};
    for (const id of ids) { const role = splits.roles[id]; if (role) counts[role] = (counts[role] ?? 0) + 1; }
    return (Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] as SplitRole | undefined) ?? null;
  };
  return <details className="video-roles">
    <summary>{t.title}</summary>
    <p className="split-note">{t.intro}</p>
    {[...videos.entries()].map(([videoID, video]) => {
      const role = current(video.ids);
      return <label key={videoID} className="video-role" data-hint={t.hint}>
        <span>{video.name}<small> · {video.ids.length} {t.images}{role ? ` · ${t.now}: ${names[role]}` : ''}</small></span>
        <select value={(videoRoles?.[videoID] as SplitRole | undefined) ?? 'auto'} disabled={busy} onChange={(event) => onChange(videoID, event.target.value as SplitRole | 'auto')}>
          <option value="auto">{t.auto}</option>
          {(['train', 'valid', 'test'] as SplitRole[]).map((value) => <option key={value} value={value}>{names[value]}</option>)}
        </select>
      </label>;
    })}
  </details>;
}

export const splitText = TEXT;

export function roleOf(splits: SplitPreview | null | undefined, frameId: string): SplitRole | null {
  return splits?.roles[frameId] ?? null;
}

export function filterFrames<T extends { id: string }>(frames: T[], splits: SplitPreview | null | undefined, filter: SplitFilterValue): T[] {
  if (!splits || filter === 'all') return frames;
  return frames.filter((frame) => filter === 'none' ? !splits.roles[frame.id] : splits.roles[frame.id] === filter);
}

/** Chips above the image list: how many images train, validate and test, as a filter. */
export function SplitFilter({ language, splits, frameIds, value, onChange }: { language: Language; splits: SplitPreview | null | undefined; frameIds: string[]; value: SplitFilterValue; onChange: (value: SplitFilterValue) => void }) {
  if (!splits) return null;
  const t = TEXT[language];
  const none = frameIds.filter((id) => !splits.roles[id]).length;
  const options: Array<[SplitFilterValue, string, number, string]> = [
    ['all', t.all, frameIds.length, t.hint],
    ['train', t.roles.train, splits.counts.train, t.hint],
    ['valid', t.roles.valid, splits.counts.valid, t.hint],
    ['test', t.roles.test, splits.counts.test, t.hint],
    ['none', t.none, none, t.noneHint],
  ];
  return <div className="split-filter">
    <div className="split-chips" role="group" aria-label={t.summary(splits.counts)}>
      {options.map(([key, label, count, hint]) => <button data-hint={hint} type="button" key={key} className={`split-chip split-${key}${value === key ? ' active' : ''}`} aria-pressed={value === key} onClick={() => onChange(key)}>{label}<strong>{count}</strong></button>)}
    </div>
    {splits.problems.map((problem) => <p key={problem} className="split-problem">{problem}</p>)}
  </div>;
}

/** The split the next training uses, for the training dialog. */
export function SplitSummary({ language, splits }: { language: Language; splits: SplitPreview | null | undefined }) {
  if (!splits) return null;
  const t = TEXT[language];
  return <div className="split-summary">
    <p><strong>{t.summary(splits.counts)}</strong></p>
    <p className="split-note">{splits.warning || t.perVideo}</p>
    {splits.problems.map((problem) => <p key={problem} className="split-problem">{problem}</p>)}
  </div>;
}
