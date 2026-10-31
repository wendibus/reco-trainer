import { useEffect, useState } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';
type HubSize = 'nano' | 'small' | 'medium';
const SIZES: HubSize[] = ['nano', 'small', 'medium'];
const SIZE_NAMES: Record<HubSize, string> = { nano: 'Nano', small: 'Small', medium: 'Medium' };

type HubModel = {
  id: string; name: string; version: string; size: HubSize; author?: string; description?: string; classes?: string[]; parent?: string | null; classMatch?: 'same' | 'different';
  training?: { images?: number; games?: number; camera?: string; venue?: string }; metrics?: { self_reported?: { mAP50?: number | null; validation_images?: number | null } }; files?: Array<{ size_bytes?: number }>;
};
type HubState = { models: HubModel[]; offline?: boolean; notYetPublished?: boolean; comparison?: { before: number; after: number; validationImages: number; keptParent: boolean; parent: string } | null };
export type HubShare = { name: string; description: string; version: string; camera: string; venue: 'indoor' | 'outdoor'; games: number; rightsConfirmed: true };

export const hubLabels = {
  de: {
    open: 'Model Hub', openHint: 'Zeigt fertige Modelle anderer Nutzer für diese Sportart. Erst dieser Klick geht online.', empty: 'Noch kein Modell? Im Model Hub gibt es vielleicht einen guten Startpunkt für diese Sportart.',
    share: 'Im Hub teilen', shareHint: 'Lädt nur die bereinigten Gewichte hoch und öffnet einen Pull Request. Bilder bleiben auf diesem Rechner.',
    title: 'Model Hub', intro: 'Modelle als Startpunkt für dein eigenes Training. Die Werte sind Angaben der Autoren auf deren eigenen Bildern und nicht vergleichbar. Beim Training misst der Trainer vorher und nachher auf deinen Bildern.',
    loading: 'Lade Liste …', offline: 'Keine Verbindung. Die Liste ist die letzte gespeicherte Kopie.', offlineEmpty: 'Keine Verbindung zum Hub.', none: 'Für diese Sportart und Größe gibt es noch kein Modell.', notYet: 'Der Hub ist noch nicht veröffentlicht.',
    use: 'Laden und aktivieren', close: 'Schließen', by: 'von', images: 'Bilder', games: 'Spiele', score: 'Selbstauskunft mAP50', on: 'auf', valImages: 'Validierungsbildern', based: 'Basiert auf',
    different: 'Andere Klassen als dein Projekt: Beim Training startet der Erkennungskopf neu, nur das Grundgerüst hilft.', indoor: 'Halle', outdoor: 'Draußen',
    comparison: (before: string, after: string, images: number, kept: boolean) => `Letztes Training auf ${images} eigenen Validierungsbildern: vorher ${before}, nachher ${after}.${kept ? ' Das Hub-Modell blieb aktiv.' : ''}`,
    shareTitle: 'Modell im Hub teilen', shareIntro: 'Hochgeladen werden nur die bereinigten Gewichte und diese Angaben, öffentlich und dauerhaft. Bilder, Videos und lokale Pfade bleiben auf diesem Rechner. Du meldest dich dafür bei GitHub an.',
    name: 'Name', namePlaceholder: 'z. B. Halle Kunstlicht Vollfeld', nameHint: 'Keine Vereine, Spieler oder Kinder nennen.', description: 'Beschreibung', version: 'Version', camera: 'Kamera', cameraPlaceholder: 'z. B. feste Kamera, Vollfeld', venue: 'Umgebung', gameCount: 'Anzahl Spiele oder Hallen im Training',
    rights: 'Ich habe die Rechte an diesem Modell und an den Daten, mit denen es trainiert wurde. Die abgebildeten Personen, bei Minderjährigen die Erziehungsberechtigten, haben zugestimmt, soweit das nötig ist. Ich trage die Verantwortung für die Veröffentlichung.',
    upload: 'Hochladen', cancel: 'Abbrechen', notReady: 'Der GitHub-Login für den Hub ist noch nicht eingerichtet.',
  },
  en: {
    open: 'Model Hub', openHint: 'Shows ready-made models from other users for this sport. Only this click goes online.', empty: 'No model yet? The Model Hub may have a good starting point for this sport.',
    share: 'Share in hub', shareHint: 'Uploads only the cleaned weights and opens a pull request. Pictures stay on this computer.',
    title: 'Model Hub', intro: 'Models as a starting point for your own training. Scores are the authors’ own, on their own images, and not comparable. When training, the trainer measures before and after on your images.',
    loading: 'Loading list …', offline: 'No connection. This list is the last saved copy.', offlineEmpty: 'No connection to the hub.', none: 'There is no model for this sport and size yet.', notYet: 'The hub has not been published yet.',
    use: 'Load and activate', close: 'Close', by: 'by', images: 'images', games: 'games', score: 'Self-reported mAP50', on: 'on', valImages: 'validation images', based: 'Based on',
    different: 'Different classes than your project: at training the detection head starts over and only the backbone helps.', indoor: 'Indoor', outdoor: 'Outdoor',
    comparison: (before: string, after: string, images: number, kept: boolean) => `Last training on ${images} of your own validation images: before ${before}, after ${after}.${kept ? ' The hub model stayed active.' : ''}`,
    shareTitle: 'Share a model in the hub', shareIntro: 'Only the cleaned weights and these details are uploaded, publicly and permanently. Pictures, videos and local paths stay on this computer. You sign in to GitHub for this.',
    name: 'Name', namePlaceholder: 'e.g. Indoor artificial light full court', nameHint: 'Do not name clubs, players or children.', description: 'Description', version: 'Version', camera: 'Camera', cameraPlaceholder: 'e.g. fixed camera, full court', venue: 'Environment', gameCount: 'Number of games or venues in training',
    rights: 'I hold the rights to this model and to the data it was trained on. The people shown, or the guardians of minors, have agreed where that is needed. I am responsible for this publication.',
    upload: 'Upload', cancel: 'Cancel', notReady: 'The GitHub login for the hub has not been set up yet.',
  },
  es: {
    open: 'Model Hub', openHint: 'Muestra modelos listos de otros usuarios para este deporte. Solo este clic se conecta a internet.', empty: '¿Aún sin modelo? En el Model Hub quizá haya un buen punto de partida para este deporte.',
    share: 'Compartir en el hub', shareHint: 'Sube solo los pesos limpios y abre un pull request. Las imágenes permanecen en este equipo.',
    title: 'Model Hub', intro: 'Modelos como punto de partida para tu propio entrenamiento. Los valores son de los autores, en sus propias imágenes, y no son comparables. Al entrenar se mide antes y después en tus imágenes.',
    loading: 'Cargando lista …', offline: 'Sin conexión. Esta lista es la última copia guardada.', offlineEmpty: 'Sin conexión con el hub.', none: 'Aún no hay ningún modelo para este deporte y tamaño.', notYet: 'El hub aún no está publicado.',
    use: 'Cargar y activar', close: 'Cerrar', by: 'de', images: 'imágenes', games: 'partidos', score: 'mAP50 declarado', on: 'en', valImages: 'imágenes de validación', based: 'Basado en',
    different: 'Clases distintas a las de tu proyecto: al entrenar, el cabezal de detección empieza de nuevo y solo ayuda la red base.', indoor: 'Interior', outdoor: 'Exterior',
    comparison: (before: string, after: string, images: number, kept: boolean) => `Último entrenamiento en ${images} imágenes de validación propias: antes ${before}, después ${after}.${kept ? ' El modelo del hub siguió activo.' : ''}`,
    shareTitle: 'Compartir un modelo en el hub', shareIntro: 'Solo se suben los pesos limpios y estos datos, de forma pública y permanente. Las imágenes, los vídeos y las rutas locales permanecen en este equipo. Para ello inicias sesión en GitHub.',
    name: 'Nombre', namePlaceholder: 'p. ej. Pabellón luz artificial pista completa', nameHint: 'No nombres clubes, jugadores ni niños.', description: 'Descripción', version: 'Versión', camera: 'Cámara', cameraPlaceholder: 'p. ej. cámara fija, pista completa', venue: 'Entorno', gameCount: 'Número de partidos o sedes en el entrenamiento',
    rights: 'Tengo los derechos sobre este modelo y sobre los datos con los que se entrenó. Las personas que aparecen, o los tutores de los menores, han dado su consentimiento cuando es necesario. Soy responsable de esta publicación.',
    upload: 'Subir', cancel: 'Cancelar', notReady: 'El inicio de sesión de GitHub para el hub aún no está configurado.',
  },
  fr: {
    open: 'Model Hub', openHint: 'Affiche des modèles prêts d’autres utilisateurs pour ce sport. Seul ce clic va en ligne.', empty: 'Pas encore de modèle ? Le Model Hub propose peut-être un bon point de départ pour ce sport.',
    share: 'Partager dans le hub', shareHint: 'Téléverse uniquement les poids nettoyés et ouvre une pull request. Les images restent sur cet ordinateur.',
    title: 'Model Hub', intro: 'Des modèles comme point de départ pour votre propre entraînement. Les scores sont ceux des auteurs, sur leurs propres images, et ne sont pas comparables. À l’entraînement, l’avant et l’après sont mesurés sur vos images.',
    loading: 'Chargement de la liste …', offline: 'Pas de connexion. Cette liste est la dernière copie enregistrée.', offlineEmpty: 'Pas de connexion au hub.', none: 'Il n’y a pas encore de modèle pour ce sport et cette taille.', notYet: 'Le hub n’est pas encore publié.',
    use: 'Charger et activer', close: 'Fermer', by: 'par', images: 'images', games: 'matchs', score: 'mAP50 déclaré', on: 'sur', valImages: 'images de validation', based: 'Basé sur',
    different: 'Classes différentes de votre projet : à l’entraînement, la tête de détection repart de zéro et seul le réseau de base aide.', indoor: 'Intérieur', outdoor: 'Extérieur',
    comparison: (before: string, after: string, images: number, kept: boolean) => `Dernier entraînement sur ${images} de vos images de validation : avant ${before}, après ${after}.${kept ? ' Le modèle du hub est resté actif.' : ''}`,
    shareTitle: 'Partager un modèle dans le hub', shareIntro: 'Seuls les poids nettoyés et ces informations sont téléversés, publiquement et durablement. Les images, les vidéos et les chemins locaux restent sur cet ordinateur. Vous vous connectez à GitHub pour cela.',
    name: 'Nom', namePlaceholder: 'p. ex. Salle lumière artificielle terrain entier', nameHint: 'Ne nommez ni clubs, ni joueurs, ni enfants.', description: 'Description', version: 'Version', camera: 'Caméra', cameraPlaceholder: 'p. ex. caméra fixe, terrain entier', venue: 'Environnement', gameCount: 'Nombre de matchs ou de salles à l’entraînement',
    rights: 'Je détiens les droits sur ce modèle et sur les données d’entraînement. Les personnes représentées, ou les tuteurs des mineurs, ont donné leur accord lorsque c’est nécessaire. Je suis responsable de cette publication.',
    upload: 'Téléverser', cancel: 'Annuler', notReady: 'La connexion GitHub du hub n’est pas encore configurée.',
  },
};

const percent = (value: number) => `${Math.round(value * 1000) / 10} %`;

export function HubDialog({ language, api, version, current, busy, onClose, onInstall }: { language: Language; api: string; version: string; current: string; busy: boolean; onClose: () => void; onInstall: (size: HubSize, id: string) => void }) {
  const t = hubLabels[language];
  const [state, setState] = useState<HubState | null>(null);
  const [size, setSize] = useState<HubSize>(SIZES.includes(current as HubSize) ? current as HubSize : 'small');
  useEffect(() => {
    let live = true;
    void fetch(`${api}/api/hub/models?version=${encodeURIComponent(version)}`).then((response) => response.json()).then((next: HubState) => { if (live) setState(next); }).catch(() => { if (live) setState({ models: [], offline: true }); });
    return () => { live = false; };
  }, [api, version]);
  const models = (state?.models ?? []).filter((model) => model.size === size);
  return <div className="walkthrough-backdrop train-backdrop"><div className="train-dialog hub-dialog" role="dialog" aria-modal="true" aria-label={t.title}>
    <h2>{t.title}</h2>
    <p className="train-note">{t.intro}</p>
    {state?.comparison && <p className="train-note">{t.comparison(percent(state.comparison.before), percent(state.comparison.after), state.comparison.validationImages, state.comparison.keptParent)}</p>}
    <div className="hub-tabs">{SIZES.map((item) => <button data-hint={SIZE_NAMES[item]} key={item} type="button" className="secondary-button" aria-pressed={item === size} onClick={() => setSize(item)}>{SIZE_NAMES[item]}</button>)}</div>
    {!state && <p>{t.loading}</p>}
    {state?.offline && <p className="train-note">{state.models.length ? t.offline : t.offlineEmpty}</p>}
    {state?.notYetPublished && <p className="train-note">{t.notYet}</p>}
    {state && !state.offline && !state.notYetPublished && !models.length && <p className="train-note">{t.none}</p>}
    {models.map((model) => {
      const own = model.metrics?.self_reported;
      return <article className="hub-model" key={model.id}><div>
        <b>{model.name} {model.version}</b>
        <small>{t.by} {model.author} · {(model.classes ?? []).join(', ')}{model.files?.[0]?.size_bytes ? ` · ${Math.round(model.files[0].size_bytes / 1e6)} MB` : ''}</small>
        <small>{model.training?.images ?? 0} {t.images} · {model.training?.games ?? 0} {t.games} · {model.training?.camera} · {model.training?.venue === 'outdoor' ? t.outdoor : t.indoor}</small>
        {own?.mAP50 != null && <small>{t.score}: {percent(own.mAP50)}{own.validation_images ? ` ${t.on} ${own.validation_images} ${t.valImages}` : ''}</small>}
        {model.parent && <small>{t.based}: {model.parent}</small>}
        {model.description && <p>{model.description}</p>}
        {model.classMatch === 'different' && <p className="train-note">{t.different}</p>}
      </div><button data-hint={t.use} type="button" className="primary-button" disabled={busy} onClick={() => onInstall(model.size, model.id)}>{t.use}</button></article>;
    })}
    <div className="train-actions"><button data-hint={t.close} type="button" className="secondary-button" onClick={onClose}>{t.close}</button></div>
  </div></div>;
}

export function HubShareDialog({ language, onCancel, onShare }: { language: Language; onCancel: () => void; onShare: (details: HubShare) => void }) {
  const t = hubLabels[language];
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [version, setVersion] = useState('1.0.0');
  const [camera, setCamera] = useState('');
  const [venue, setVenue] = useState<'indoor' | 'outdoor'>('indoor');
  const [games, setGames] = useState(1);
  const [rights, setRights] = useState(false);
  const ready = name.trim().length >= 2 && camera.trim().length > 0 && games >= 1 && /^\d+\.\d+\.\d+$/.test(version) && rights;
  return <div className="walkthrough-backdrop train-backdrop"><div className="train-dialog hub-dialog" role="dialog" aria-modal="true" aria-label={t.shareTitle}>
    <h2>{t.shareTitle}</h2>
    <p className="train-note">{t.shareIntro}</p>
    <fieldset className="hub-form">
      <label>{t.name}<input type="text" maxLength={60} placeholder={t.namePlaceholder} value={name} onChange={(event) => setName(event.target.value)} /></label>
      <p className="train-note">{t.nameHint}</p>
      <label>{t.description}<textarea rows={2} maxLength={400} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      <label>{t.version}<input type="text" value={version} onChange={(event) => setVersion(event.target.value)} /></label>
      <label>{t.camera}<input type="text" maxLength={80} placeholder={t.cameraPlaceholder} value={camera} onChange={(event) => setCamera(event.target.value)} /></label>
      <label>{t.venue}<select value={venue} onChange={(event) => setVenue(event.target.value as 'indoor' | 'outdoor')}><option value="indoor">{t.indoor}</option><option value="outdoor">{t.outdoor}</option></select></label>
      <label>{t.gameCount}<input type="number" min={1} max={10000} value={games} onChange={(event) => setGames(Math.max(1, Number(event.target.value) || 1))} /></label>
      <label className="hub-rights"><input type="checkbox" checked={rights} onChange={(event) => setRights(event.target.checked)} /> <span>{t.rights}</span></label>
    </fieldset>
    <div className="train-actions">
      <button data-hint={t.cancel} type="button" className="secondary-button" onClick={onCancel}>{t.cancel}</button>
      <button data-hint={t.upload} type="button" className="primary-button" disabled={!ready} onClick={() => onShare({ name: name.trim(), description: description.trim(), version, camera: camera.trim(), venue, games, rightsConfirmed: true })}>{t.upload}</button>
    </div>
  </div></div>;
}
