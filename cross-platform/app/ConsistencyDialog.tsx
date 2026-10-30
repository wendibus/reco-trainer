import { useState } from 'react';
import type { Job } from './JobDialog';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type Tier = 'strong' | 'medium' | 'weak';
type RegistryModel = { id: string; bytes?: number; license?: string; installed?: boolean | null };
export type ConsistencyOptions = { iou: number; vlm: boolean; vlmModel: 'qwen3.5-4b-vlm' | 'qwen3.5-9b-vlm'; sample: number };

const text: Record<Language, {
  title: string; intro: string; iou: string; iouHint: string; vlm: string; vlmHint: string; vlmWeak: string; model: string; model9b: string;
  sample: string; sampleHint: string; download: (size: string) => string; downloaded: string; start: string; cancel: string;
  result: (checked: number, flagged: number) => string; details: (disagree: number, vanished: number) => string; vlmResult: (checked: number, issues: number, cleared: number) => string;
  hard: (count: number) => string; open: (index: number) => string; none: string;
}> = {
  de: {
    title: 'Konsistenz prüfen', intro: 'Ein zweiter Detektor und der zeitliche Verlauf suchen nach fehlenden, doppelten oder springenden Boxen. Auf Wunsch schaut sich ein lokales Sprach-Bild-Modell nur die auffälligen Bilder und eine Stichprobe an.',
    iou: 'Übereinstimmung ab IoU', iouHint: 'Ab diesem Überlappungsmaß gelten zwei Boxen als dieselbe. Höher = strenger.',
    vlm: 'Lokales VLM (Qwen3.5) prüfen lassen', vlmHint: 'Läuft vollständig auf diesem Rechner. Bestätigte Probleme werden als „Hard Example“ markiert und im Training doppelt verwendet; unbegründete Warnungen werden entfernt.',
    vlmWeak: 'Auf dieser Hardware-Stufe (nur CPU) wird das VLM nicht angeboten. Die Prüfung ohne VLM funktioniert trotzdem.',
    model: 'Modell', model9b: 'Qwen3.5-9B (genauer, braucht viel Speicher)', sample: 'Zusätzliche Stichprobe', sampleHint: 'Anteil der unauffälligen Bilder, die das VLM ebenfalls ansieht.',
    download: (size) => `Beim ersten Mal werden etwa ${size} geladen (einmal pro Rechner).`, downloaded: 'Bereits geladen.', start: 'Prüfen', cancel: 'Abbrechen',
    result: (checked, flagged) => `${checked} Bilder geprüft, ${flagged} zur Prüfung markiert.`, details: (disagree, vanished) => `Zweiter Detektor uneinig: ${disagree} · plötzlich fehlende Spieler: ${vanished}`,
    vlmResult: (checked, issues, cleared) => `VLM: ${checked} angesehen, ${issues} beanstandet (Hard Examples), ${cleared} entwarnt.`, hard: (count) => `${count} Hard Examples insgesamt.`,
    open: (index) => `Bild ${index}`, none: 'Keine Auffälligkeiten.',
  },
  en: {
    title: 'Check consistency', intro: 'A second detector and the timeline look for missing, duplicate or jumping boxes. Optionally, a local vision-language model looks at the suspicious images and a sample only.',
    iou: 'Match from IoU', iouHint: 'From this overlap on, two boxes count as the same object. Higher = stricter.',
    vlm: 'Let the local VLM (Qwen3.5) check', vlmHint: 'Runs entirely on this computer. Confirmed problems are marked as hard examples and used twice in training; unfounded warnings are removed.',
    vlmWeak: 'On this hardware tier (CPU only) the VLM is not offered. The check without the VLM still works.',
    model: 'Model', model9b: 'Qwen3.5-9B (more accurate, needs a lot of memory)', sample: 'Additional sample', sampleHint: 'Share of unremarkable images the VLM also looks at.',
    download: (size) => `About ${size} are downloaded the first time (once per computer).`, downloaded: 'Already downloaded.', start: 'Check', cancel: 'Cancel',
    result: (checked, flagged) => `${checked} images checked, ${flagged} flagged for review.`, details: (disagree, vanished) => `Second detector disagrees: ${disagree} · players suddenly missing: ${vanished}`,
    vlmResult: (checked, issues, cleared) => `VLM: ${checked} looked at, ${issues} with issues (hard examples), ${cleared} cleared.`, hard: (count) => `${count} hard examples in total.`,
    open: (index) => `Image ${index}`, none: 'Nothing suspicious.',
  },
  es: {
    title: 'Comprobar coherencia', intro: 'Un segundo detector y la línea temporal buscan cuadros que faltan, duplicados o que saltan. Opcionalmente, un modelo de visión y lenguaje local revisa solo las imágenes sospechosas y una muestra.',
    iou: 'Coincidencia desde IoU', iouHint: 'A partir de este solapamiento, dos cuadros cuentan como el mismo objeto. Más alto = más estricto.',
    vlm: 'Revisar con el VLM local (Qwen3.5)', vlmHint: 'Funciona por completo en este equipo. Los problemas confirmados se marcan como ejemplos difíciles y se usan dos veces en el entrenamiento; los avisos infundados se eliminan.',
    vlmWeak: 'En este nivel de hardware (solo CPU) no se ofrece el VLM. La comprobación sin VLM sigue funcionando.',
    model: 'Modelo', model9b: 'Qwen3.5-9B (más preciso, necesita mucha memoria)', sample: 'Muestra adicional', sampleHint: 'Parte de las imágenes sin incidencias que el VLM también revisa.',
    download: (size) => `La primera vez se descargan unos ${size} (una vez por equipo).`, downloaded: 'Ya descargado.', start: 'Comprobar', cancel: 'Cancelar',
    result: (checked, flagged) => `${checked} imágenes comprobadas, ${flagged} marcadas para revisar.`, details: (disagree, vanished) => `Segundo detector en desacuerdo: ${disagree} · jugadores que desaparecen: ${vanished}`,
    vlmResult: (checked, issues, cleared) => `VLM: ${checked} revisadas, ${issues} con problemas (ejemplos difíciles), ${cleared} descartadas.`, hard: (count) => `${count} ejemplos difíciles en total.`,
    open: (index) => `Imagen ${index}`, none: 'Nada sospechoso.',
  },
  fr: {
    title: 'Vérifier la cohérence', intro: 'Un second détecteur et la chronologie cherchent des boîtes manquantes, en double ou qui sautent. En option, un modèle vision-langage local n’examine que les images suspectes et un échantillon.',
    iou: 'Correspondance à partir d’un IoU de', iouHint: 'À partir de ce recouvrement, deux boîtes désignent le même objet. Plus élevé = plus strict.',
    vlm: 'Faire vérifier par le VLM local (Qwen3.5)', vlmHint: 'Fonctionne entièrement sur cet ordinateur. Les problèmes confirmés sont marqués comme exemples difficiles et utilisés deux fois à l’entraînement ; les alertes infondées sont retirées.',
    vlmWeak: 'Sur ce niveau matériel (CPU uniquement), le VLM n’est pas proposé. La vérification sans VLM fonctionne quand même.',
    model: 'Modèle', model9b: 'Qwen3.5-9B (plus précis, demande beaucoup de mémoire)', sample: 'Échantillon supplémentaire', sampleHint: 'Part des images sans anomalie que le VLM examine aussi.',
    download: (size) => `Environ ${size} sont téléchargés la première fois (une fois par ordinateur).`, downloaded: 'Déjà téléchargé.', start: 'Vérifier', cancel: 'Annuler',
    result: (checked, flagged) => `${checked} images vérifiées, ${flagged} signalées pour vérification.`, details: (disagree, vanished) => `Second détecteur en désaccord : ${disagree} · joueurs disparus soudainement : ${vanished}`,
    vlmResult: (checked, issues, cleared) => `VLM : ${checked} examinées, ${issues} avec problèmes (exemples difficiles), ${cleared} levées.`, hard: (count) => `${count} exemples difficiles au total.`,
    open: (index) => `Image ${index}`, none: 'Rien de suspect.',
  },
};

const sizeLabel = (bytes: number) => bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;

export function ConsistencyDialog({ language, tier, registry, onStart, onClose }: {
  language: Language; tier: Tier; registry: RegistryModel[]; onStart: (options: ConsistencyOptions) => void; onClose: () => void;
}) {
  const t = text[language];
  const vlmAllowed = tier !== 'weak';
  const [options, setOptions] = useState<ConsistencyOptions>({ iou: 0.5, vlm: tier === 'strong', vlmModel: 'qwen3.5-4b-vlm', sample: 0.05 });
  const model = registry.find((entry) => entry.id === options.vlmModel);
  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <form className="onnx-dialog export-dialog" role="dialog" aria-modal="true" aria-labelledby="consistency-title" onSubmit={(event) => { event.preventDefault(); onStart({ ...options, vlm: options.vlm && vlmAllowed }); }}>
      <h2 id="consistency-title">{t.title}</h2>
      <p>{t.intro}</p>
      <label>{t.iou} {options.iou.toFixed(2)}<input type="range" min={0.3} max={0.9} step={0.05} value={options.iou} onChange={(event) => setOptions({ ...options, iou: Number(event.target.value) })} /><small>{t.iouHint}</small></label>
      <label className="export-toggle"><input type="checkbox" checked={options.vlm && vlmAllowed} disabled={!vlmAllowed} onChange={(event) => setOptions({ ...options, vlm: event.target.checked })} /><span><strong>{t.vlm}</strong><small>{vlmAllowed ? t.vlmHint : t.vlmWeak}</small></span></label>
      {options.vlm && vlmAllowed && <div className="export-suboptions">
        <label>{t.model}<select value={options.vlmModel} onChange={(event) => setOptions({ ...options, vlmModel: event.target.value as ConsistencyOptions['vlmModel'] })}>
          <option value="qwen3.5-4b-vlm">Qwen3.5-4B</option>
          {tier === 'strong' && <option value="qwen3.5-9b-vlm">{t.model9b}</option>}
        </select><small>{model?.installed ? t.downloaded : t.download(sizeLabel(model?.bytes ?? 3.4e9))} · {model?.license ?? 'Apache-2.0'}</small></label>
        <label>{t.sample} {Math.round(options.sample * 100)} %<input type="range" min={0} max={0.5} step={0.05} value={options.sample} onChange={(event) => setOptions({ ...options, sample: Number(event.target.value) })} /><small>{t.sampleHint}</small></label>
      </div>}
      <div className="onnx-actions"><button data-hint={hintText[language].cancel} type="button" className="secondary-button" onClick={onClose}>{t.cancel}</button><button data-hint={hintText[language].startCheck} type="submit" className="primary-button">{t.start}</button></div>
    </form>
  </div>;
}

export function ConsistencyResult({ job, language, onOpenFrame }: { job: Job; language: Language; onOpenFrame: (frameID: string) => void }) {
  const t = text[language];
  const result = job.result as { kind?: string; checked?: number; review?: string[]; detectorDisagreement?: number; playersVanished?: number; vlmChecked?: number; vlmIssues?: number; vlmResolved?: number; hardExamples?: number } | null | undefined;
  if (!result || result.kind !== 'consistency') return null;
  const review = result.review ?? [];
  return <div className="export-result">
    <p><strong>{t.result(result.checked ?? 0, review.length)}</strong></p>
    <p>{t.details(result.detectorDisagreement ?? 0, result.playersVanished ?? 0)}</p>
    {(result.vlmChecked ?? 0) > 0 && <p>{t.vlmResult(result.vlmChecked ?? 0, result.vlmIssues ?? 0, result.vlmResolved ?? 0)}</p>}
    {(result.hardExamples ?? 0) > 0 && <p>{t.hard(result.hardExamples ?? 0)}</p>}
    {review.length ? <div className="export-held-list">{review.slice(0, 60).map((frameID, index) => <button data-hint={hintText[language].openFrame} type="button" key={frameID} className="tool-button" onClick={() => onOpenFrame(frameID)}>{t.open(index + 1)}</button>)}</div> : <p>{t.none}</p>}
  </div>;
}
