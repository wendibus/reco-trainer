import { useState } from 'react';
import type { Job } from './JobDialog';

type Language = 'de' | 'en' | 'es' | 'fr';
export type RegistryModel = { id: string; name: string; purpose?: string; license: string; licenseUrl?: string; source?: string; installed?: boolean | null };
export type ExportOptions = { formats: Array<'yolo' | 'coco'>; anonymize: boolean; method: 'gaussian' | 'pixelate'; detectPeople: boolean; includeText: boolean; zip: boolean };

const text: Record<Language, {
  title: string; intro: string; format: string; both: string; anonymize: string; anonymizeHint: string; needsMl: string;
  method: string; gaussian: string; pixelate: string; people: string; peopleHint: string; textOption: string; textHint: string;
  zip: string; models: string; start: string; cancel: string;
  resultImages: (images: number, boxes: number) => string; resultFolder: string; showFolder: string; heldBack: (count: number) => string; heldBackHint: string; openFrame: (index: number) => string;
}> = {
  de: {
    title: 'Datensatz exportieren', intro: 'Exportiert alle geprüften Bilder mit ihren Labels, aufgeteilt nach Video in Training, Validierung und Test. Dateinamen verraten keine Videos oder Ordner.',
    format: 'Format', both: 'beide', anonymize: 'Anonymisieren', anonymizeHint: 'Köpfe und Gesichter werden stark weichgezeichnet, nicht abgedeckt – die Kopfform bleibt, damit sich weiter sinnvoll trainieren lässt. Labels bleiben unverändert.',
    needsMl: 'Für die Anonymisierung zuerst „ML einrichten“ anklicken.', method: 'Methode', gaussian: 'Weichzeichnen', pixelate: 'Verpixeln',
    people: 'Zuschauer, Bank und weitere Personen finden', peopleHint: 'Findet auch Personen ohne Label mit dem RF-DETR-Basismodell (COCO).',
    textOption: 'Auch Trikotnummern und Anzeigetafel unkenntlich machen', textHint: 'Schriftzüge werden erkannt und weichgezeichnet. Kostet etwas Zeit.',
    zip: 'ZIP-Datei erstellen', models: 'Verwendete Modelle und Lizenzen', start: 'Exportieren', cancel: 'Abbrechen',
    resultImages: (images, boxes) => `${images} Bilder, ${boxes} Boxen.`, resultFolder: 'Gespeichert in', showFolder: 'Ordner anzeigen',
    heldBack: (count) => `${count} ${count === 1 ? 'Bild wurde' : 'Bilder wurden'} nicht exportiert, weil dort nach der Anonymisierung noch ein Gesicht erkennbar war.`,
    heldBackHint: 'Bitte ansehen und bei Bedarf die Personen markieren; danach erneut exportieren.', openFrame: (index) => `Bild ${index} öffnen`,
  },
  en: {
    title: 'Export dataset', intro: 'Exports every reviewed image with its labels, split by video into training, validation and test. File names reveal no videos or folders.',
    format: 'Format', both: 'both', anonymize: 'Anonymize', anonymizeHint: 'Heads and faces are strongly blurred, not covered – the head shape stays so the data remains useful for training. Labels are unchanged.',
    needsMl: 'Click “Set up ML” first to anonymize.', method: 'Method', gaussian: 'Blur', pixelate: 'Pixelate',
    people: 'Find spectators, bench and other people', peopleHint: 'Also finds people without a label, using the RF-DETR base model (COCO).',
    textOption: 'Also hide jersey numbers and the scoreboard', textHint: 'Text is detected and blurred. Takes a little longer.',
    zip: 'Create a ZIP file', models: 'Models used and their licenses', start: 'Export', cancel: 'Cancel',
    resultImages: (images, boxes) => `${images} images, ${boxes} boxes.`, resultFolder: 'Saved in', showFolder: 'Show folder',
    heldBack: (count) => `${count} ${count === 1 ? 'image was' : 'images were'} not exported because a face was still detectable after anonymization.`,
    heldBackHint: 'Please look at them and label the people if needed, then export again.', openFrame: (index) => `Open image ${index}`,
  },
  es: {
    title: 'Exportar conjunto de datos', intro: 'Exporta todas las imágenes revisadas con sus etiquetas, separadas por vídeo en entrenamiento, validación y prueba. Los nombres de archivo no revelan vídeos ni carpetas.',
    format: 'Formato', both: 'ambos', anonymize: 'Anonimizar', anonymizeHint: 'Las cabezas y caras se difuminan fuertemente, sin taparlas: la forma de la cabeza se mantiene para que los datos sigan sirviendo para entrenar. Las etiquetas no cambian.',
    needsMl: 'Para anonimizar, primero haz clic en «Configurar ML».', method: 'Método', gaussian: 'Difuminar', pixelate: 'Pixelar',
    people: 'Encontrar público, banquillo y otras personas', peopleHint: 'También encuentra personas sin etiqueta con el modelo base RF-DETR (COCO).',
    textOption: 'Ocultar también dorsales y marcador', textHint: 'El texto se detecta y se difumina. Tarda un poco más.',
    zip: 'Crear un archivo ZIP', models: 'Modelos usados y sus licencias', start: 'Exportar', cancel: 'Cancelar',
    resultImages: (images, boxes) => `${images} imágenes, ${boxes} cuadros.`, resultFolder: 'Guardado en', showFolder: 'Mostrar carpeta',
    heldBack: (count) => `${count} ${count === 1 ? 'imagen no se exportó' : 'imágenes no se exportaron'} porque tras la anonimización aún se detectaba una cara.`,
    heldBackHint: 'Revísalas y marca a las personas si hace falta; luego exporta de nuevo.', openFrame: (index) => `Abrir imagen ${index}`,
  },
  fr: {
    title: 'Exporter le jeu de données', intro: 'Exporte toutes les images vérifiées avec leurs annotations, réparties par vidéo en entraînement, validation et test. Les noms de fichiers ne révèlent ni vidéos ni dossiers.',
    format: 'Format', both: 'les deux', anonymize: 'Anonymiser', anonymizeHint: 'Les têtes et visages sont fortement floutés, pas masqués : la forme de la tête reste, pour que les données servent encore à l’entraînement. Les annotations sont inchangées.',
    needsMl: 'Pour anonymiser, cliquez d’abord sur « Configurer le ML ».', method: 'Méthode', gaussian: 'Flouter', pixelate: 'Pixeliser',
    people: 'Trouver le public, le banc et d’autres personnes', peopleHint: 'Trouve aussi les personnes sans annotation avec le modèle de base RF-DETR (COCO).',
    textOption: 'Masquer aussi les numéros de maillot et le tableau d’affichage', textHint: 'Le texte est détecté et flouté. Un peu plus long.',
    zip: 'Créer un fichier ZIP', models: 'Modèles utilisés et leurs licences', start: 'Exporter', cancel: 'Annuler',
    resultImages: (images, boxes) => `${images} images, ${boxes} boîtes.`, resultFolder: 'Enregistré dans', showFolder: 'Afficher le dossier',
    heldBack: (count) => `${count} ${count === 1 ? 'image n’a pas été exportée' : 'images n’ont pas été exportées'} car un visage restait détectable après l’anonymisation.`,
    heldBackHint: 'Veuillez les examiner et annoter les personnes si nécessaire, puis exporter à nouveau.', openFrame: (index) => `Ouvrir l’image ${index}`,
  },
};

function modelsFor(options: ExportOptions, registry: RegistryModel[], accelerated: boolean): RegistryModel[] {
  if (!options.anonymize) return [];
  const ids = ['yunet-face', ...(options.includeText ? ['ppocr-text-detector'] : []), ...(options.detectPeople ? [accelerated ? 'rf-detr-medium' : 'rf-detr-nano'] : [])];
  return ids.map((id) => registry.find((model) => model.id === id)).filter((model): model is RegistryModel => !!model);
}

export function ExportDialog({ language, mlReady, accelerated, registry, onStart, onClose }: {
  language: Language; mlReady: boolean; accelerated: boolean; registry: RegistryModel[];
  onStart: (options: ExportOptions) => void; onClose: () => void;
}) {
  const t = text[language];
  const [options, setOptions] = useState<ExportOptions>({ formats: ['yolo', 'coco'], anonymize: mlReady, method: 'gaussian', detectPeople: true, includeText: false, zip: true });
  const set = (patch: Partial<ExportOptions>) => setOptions((current) => ({ ...current, ...patch }));
  const formatValue = options.formats.length === 2 ? 'both' : options.formats[0];
  const models = modelsFor(options, registry, accelerated);

  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <form className="onnx-dialog export-dialog" role="dialog" aria-modal="true" aria-labelledby="export-title" onSubmit={(event) => { event.preventDefault(); onStart(options); }}>
      <h2 id="export-title">{t.title}</h2>
      <p>{t.intro}</p>
      <fieldset className="export-choice"><legend>{t.format}</legend>
        {(['yolo', 'coco', 'both'] as const).map((value) => <label key={value}><input type="radio" name="format" checked={formatValue === value} onChange={() => set({ formats: value === 'both' ? ['yolo', 'coco'] : [value] })} />{value === 'both' ? t.both : value.toUpperCase()}</label>)}
      </fieldset>
      <label className="export-toggle"><input type="checkbox" checked={options.anonymize} disabled={!mlReady} onChange={(event) => set({ anonymize: event.target.checked })} /><span><strong>{t.anonymize}</strong><small>{mlReady ? t.anonymizeHint : t.needsMl}</small></span></label>
      {options.anonymize && <div className="export-suboptions">
        <fieldset className="export-choice"><legend>{t.method}</legend>
          <label><input type="radio" name="method" checked={options.method === 'gaussian'} onChange={() => set({ method: 'gaussian' })} />{t.gaussian}</label>
          <label><input type="radio" name="method" checked={options.method === 'pixelate'} onChange={() => set({ method: 'pixelate' })} />{t.pixelate}</label>
        </fieldset>
        <label className="export-toggle"><input type="checkbox" checked={options.detectPeople} onChange={(event) => set({ detectPeople: event.target.checked })} /><span>{t.people}<small>{t.peopleHint}</small></span></label>
        <label className="export-toggle"><input type="checkbox" checked={options.includeText} onChange={(event) => set({ includeText: event.target.checked })} /><span>{t.textOption}<small>{t.textHint}</small></span></label>
      </div>}
      <label className="export-toggle"><input type="checkbox" checked={options.zip} onChange={(event) => set({ zip: event.target.checked })} /><span>{t.zip}</span></label>
      {models.length > 0 && <details className="export-models"><summary>{t.models}</summary><ul>{models.map((model) => <li key={model.id}>{model.name} · {model.licenseUrl ? <a href={model.licenseUrl} target="_blank" rel="noreferrer">{model.license}</a> : model.license}</li>)}</ul></details>}
      <div className="onnx-actions"><button type="button" className="secondary-button" onClick={onClose}>{t.cancel}</button><button type="submit" className="primary-button">{t.start}</button></div>
    </form>
  </div>;
}

/** Result part of the job window after a dataset export. */
export function ExportResult({ job, language, onReveal, onOpenFrame }: {
  job: Job; language: Language; onReveal: (exportName: string) => void; onOpenFrame: (frameID: string) => void;
}) {
  const t = text[language];
  const result = job.result as { kind?: string; name?: string; folder?: string; zip?: string | null; images?: number; annotations?: number; excludedFrames?: string[]; models?: RegistryModel[] } | null | undefined;
  if (!result || result.kind !== 'dataset-export') return null;
  const held = result.excludedFrames ?? [];
  return <div className="export-result">
    <p>{t.resultImages(result.images ?? 0, result.annotations ?? 0)}</p>
    <p className="export-path"><span>{t.resultFolder}</span><code>{result.zip ?? result.folder}</code></p>
    {result.name && <button type="button" className="secondary-button" onClick={() => onReveal(result.name!)}>{t.showFolder}</button>}
    {held.length > 0 && <div className="export-held">
      <p><strong>{t.heldBack(held.length)}</strong> {t.heldBackHint}</p>
      <div className="export-held-list">{held.slice(0, 40).map((frameID, index) => <button type="button" key={frameID} className="tool-button" onClick={() => onOpenFrame(frameID)}>{t.openFrame(index + 1)}</button>)}</div>
    </div>}
    {result.models && result.models.length > 0 && <details className="export-models"><summary>{t.models}</summary><ul>{result.models.map((model) => <li key={model.id}>{model.name} · {model.license}</li>)}</ul></details>}
  </div>;
}
