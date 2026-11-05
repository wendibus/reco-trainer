import { useState } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
export type ReviewAnnotation = { id: string; category: string; x: number; y: number; width: number; height: number; source: string; confidence?: number | null };
export type ReviewFrame = {
  id: string; annotations: ReviewAnnotation[]; reviewFlags?: string[];
  vlmCheck?: { comment?: string; missing?: number; duplicates?: number; model?: string };
  consistencySuggestion?: { add?: Array<Omit<ReviewAnnotation, 'id' | 'source'>>; source?: string };
};
export type Proposal = { remove: Set<string>; add: ReviewAnnotation[]; result: ReviewAnnotation[] };

/// Boxes of the same class overlapping at least this much count as one object marked twice.
export const DUPLICATE_IOU = 0.6;

function iou(a: ReviewAnnotation, b: ReviewAnnotation) {
  const left = Math.max(a.x, b.x), top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width), bottom = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

/// The concrete change offered for a flagged frame: drop measurable duplicates (same class,
/// IoU >= DUPLICATE_IOU; the box with the lower confidence goes, hand-drawn boxes count as
/// certain, on a tie the later box goes) and add the second detector's unmatched boxes.
export function consistencyProposal(frame: ReviewFrame): Proposal {
  const remove = new Set<string>();
  const boxes = frame.annotations;
  const certainty = (box: ReviewAnnotation) => box.confidence ?? 1;
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const first = boxes[i], second = boxes[j];
      if (first.category !== second.category || remove.has(first.id) || remove.has(second.id)) continue;
      if (iou(first, second) < DUPLICATE_IOU) continue;
      remove.add(certainty(second) > certainty(first) ? first.id : second.id);
    }
  }
  const add = (frame.consistencySuggestion?.add ?? []).map((box, index) => ({ ...box, id: `suggest-${frame.id}-${index}`, source: 'manual' }));
  const result = [...boxes.filter((box) => !remove.has(box.id)), ...add.map((box) => ({ ...box, id: crypto.randomUUID() }))];
  return { remove, add, result };
}

const copy = {
  de: {
    title: 'Konsistenzprüfung', of: 'von', reasons: 'Grund', vlm: 'VLM-Kommentar', vlmCounts: (duplicates: number, missing: number) => `meldet ${duplicates} doppelt, ${missing} fehlend`,
    proposal: 'Vorschlag', removeDuplicates: (count: number) => `${count} doppelte ${count === 1 ? 'Box' : 'Boxen'} entfernen (rot)`, addBoxes: (count: number, source: string) => `${count} ${count === 1 ? 'Box' : 'Boxen'} vom zweiten Detektor ergänzen (grün, ${source})`,
    noProposal: 'Keine konkrete Änderung messbar.', falseAlarm: 'Die VLM meldet doppelte Boxen, aber keine zwei Boxen derselben Klasse überlappen stark – vermutlich Fehlalarm.',
    noDetector: 'Für dieses Bild sind noch keine Vorschläge des zweiten Detektors berechnet.', suggest: (count: number) => `Vorschläge berechnen (${count} Bilder)`,
    accept: 'Vorschlag übernehmen', edit: 'Nachbessern', editing: 'Boxen im Bild korrigieren, dann:', editDone: 'Nachgebessert, weiter', reject: 'Ablehnen, Boxen behalten', remove: 'Bild löschen', confirmRemove: 'Wirklich löschen?', skip: 'Überspringen', done: 'Alle markierten Bilder sind geprüft.',
  },
  en: {
    title: 'Consistency review', of: 'of', reasons: 'Reason', vlm: 'VLM comment', vlmCounts: (duplicates: number, missing: number) => `reports ${duplicates} duplicate, ${missing} missing`,
    proposal: 'Suggestion', removeDuplicates: (count: number) => `remove ${count} duplicate ${count === 1 ? 'box' : 'boxes'} (red)`, addBoxes: (count: number, source: string) => `add ${count} ${count === 1 ? 'box' : 'boxes'} from the second detector (green, ${source})`,
    noProposal: 'No concrete change can be measured.', falseAlarm: 'The VLM reports duplicate boxes, but no two boxes of the same class overlap strongly – probably a false alarm.',
    noDetector: 'No second-detector suggestions have been computed for this image yet.', suggest: (count: number) => `Compute suggestions (${count} images)`,
    accept: 'Accept suggestion', edit: 'Fix by hand', editing: 'Correct the boxes in the image, then:', editDone: 'Fixed, next', reject: 'Reject, keep boxes', remove: 'Delete image', confirmRemove: 'Really delete?', skip: 'Skip', done: 'All flagged images have been reviewed.',
  },
  es: {
    title: 'Revisión de coherencia', of: 'de', reasons: 'Motivo', vlm: 'Comentario del VLM', vlmCounts: (duplicates: number, missing: number) => `indica ${duplicates} duplicados, ${missing} faltantes`,
    proposal: 'Sugerencia', removeDuplicates: (count: number) => `quitar ${count} ${count === 1 ? 'cuadro duplicado' : 'cuadros duplicados'} (rojo)`, addBoxes: (count: number, source: string) => `añadir ${count} ${count === 1 ? 'cuadro' : 'cuadros'} del segundo detector (verde, ${source})`,
    noProposal: 'No se puede medir ningún cambio concreto.', falseAlarm: 'El VLM indica cuadros duplicados, pero ningún par de la misma clase se solapa mucho: probablemente una falsa alarma.',
    noDetector: 'Aún no se han calculado sugerencias del segundo detector para esta imagen.', suggest: (count: number) => `Calcular sugerencias (${count} imágenes)`,
    accept: 'Aceptar sugerencia', edit: 'Corregir a mano', editing: 'Corrige los cuadros en la imagen y luego:', editDone: 'Corregido, siguiente', reject: 'Rechazar, mantener cuadros', remove: 'Eliminar imagen', confirmRemove: '¿Eliminar de verdad?', skip: 'Omitir', done: 'Todas las imágenes marcadas están revisadas.',
  },
  fr: {
    title: 'Revue de cohérence', of: 'sur', reasons: 'Motif', vlm: 'Commentaire du VLM', vlmCounts: (duplicates: number, missing: number) => `signale ${duplicates} en double, ${missing} manquantes`,
    proposal: 'Suggestion', removeDuplicates: (count: number) => `retirer ${count} ${count === 1 ? 'boîte en double' : 'boîtes en double'} (rouge)`, addBoxes: (count: number, source: string) => `ajouter ${count} ${count === 1 ? 'boîte' : 'boîtes'} du second détecteur (vert, ${source})`,
    noProposal: 'Aucune modification concrète mesurable.', falseAlarm: 'Le VLM signale des boîtes en double, mais aucune paire de même classe ne se chevauche fortement – probablement une fausse alerte.',
    noDetector: 'Aucune suggestion du second détecteur n’a encore été calculée pour cette image.', suggest: (count: number) => `Calculer les suggestions (${count} images)`,
    accept: 'Accepter la suggestion', edit: 'Corriger à la main', editing: 'Corrigez les boîtes dans l’image, puis :', editDone: 'Corrigé, suivant', reject: 'Refuser, garder les boîtes', remove: 'Supprimer l’image', confirmRemove: 'Vraiment supprimer ?', skip: 'Passer', done: 'Toutes les images signalées ont été vérifiées.',
  },
} as const;

export const reviewListText = {
  de: { review: (count: number) => `${count} markierte Bilder prüfen`, only: 'Nur markierte zeigen' },
  en: { review: (count: number) => `Review ${count} flagged images`, only: 'Show flagged only' },
  es: { review: (count: number) => `Revisar ${count} imágenes marcadas`, only: 'Mostrar solo marcadas' },
  fr: { review: (count: number) => `Vérifier ${count} images signalées`, only: 'Afficher seulement les signalées' },
} as const;

type Props = {
  language: Language;
  frame: ReviewFrame;
  position: number;
  total: number;
  missingSuggestions: number;
  busy: boolean;
  flagLabel: (flag: string) => string;
  onDecision: (decision: 'accept' | 'reject' | 'edited', annotations?: ReviewAnnotation[]) => void;
  onDelete: () => void;
  onSkip: () => void;
  onEdit: () => void;
  onSuggest: () => void;
};

export default function ConsistencyReview(props: Props) {
  const t = copy[props.language];
  const h = hintText[props.language];
  const [editingFrame, setEditingFrame] = useState<string | null>(null);
  const [confirmFrame, setConfirmFrame] = useState<string | null>(null);
  const editing = editingFrame === props.frame.id;
  const confirming = confirmFrame === props.frame.id;
  const proposal = consistencyProposal(props.frame);
  const hasProposal = proposal.remove.size > 0 || proposal.add.length > 0;
  const vlm = props.frame.vlmCheck;
  const falseAlarm = Boolean(vlm?.duplicates) && proposal.remove.size === 0;
  const flags = props.frame.reviewFlags ?? [];

  return <section className="candidate-review consistency-review" aria-label={t.title}>
    <div className="consistency-review-text">
      <strong>{t.title}: {props.position} {t.of} {props.total}</strong>
      <small><b>{t.reasons}:</b> {flags.map(props.flagLabel).join(' · ')}</small>
      {vlm?.comment && <small className="consistency-vlm"><b>{t.vlm}</b> ({t.vlmCounts(vlm.duplicates ?? 0, vlm.missing ?? 0)}): „{vlm.comment}“</small>}
      <small><b>{t.proposal}:</b> {hasProposal ? [proposal.remove.size ? t.removeDuplicates(proposal.remove.size) : '', proposal.add.length ? t.addBoxes(proposal.add.length, props.frame.consistencySuggestion?.source ?? '') : ''].filter(Boolean).join(', ') : t.noProposal}</small>
      {falseAlarm && <small className="consistency-false-alarm">{t.falseAlarm}</small>}
      {!props.frame.consistencySuggestion && props.missingSuggestions > 0 && <small className="consistency-no-detector">{t.noDetector} <button data-hint={h.consistencySuggest} type="button" className="link-button" onClick={props.onSuggest} disabled={props.busy}>{t.suggest(props.missingSuggestions)}</button></small>}
      {editing && <small className="consistency-editing">{t.editing}</small>}
    </div>
    <div className="consistency-review-actions">
      {editing
        ? <button data-hint={h.consistencyEditDone} type="button" className="candidate-yes" onClick={() => { setEditingFrame(null); props.onDecision('edited'); }} disabled={props.busy}>✓ {t.editDone}</button>
        : <button data-hint={h.consistencyAccept} type="button" className="candidate-yes" onClick={() => props.onDecision('accept', proposal.result)} disabled={props.busy || !hasProposal}>✓ {t.accept}</button>}
      {!editing && <button data-hint={h.consistencyEdit} type="button" onClick={() => { setEditingFrame(props.frame.id); props.onEdit(); }} disabled={props.busy}>✎ {t.edit}</button>}
      <button data-hint={h.consistencyReject} type="button" className="candidate-no" onClick={() => props.onDecision('reject')} disabled={props.busy}>× {t.reject}</button>
      {confirming
        ? <button data-hint={h.removeFrame} type="button" className="consistency-delete confirm" onClick={() => { setConfirmFrame(null); props.onDelete(); }} disabled={props.busy}>⌫ {t.confirmRemove}</button>
        : <button data-hint={h.removeFrame} type="button" className="consistency-delete" onClick={() => setConfirmFrame(props.frame.id)} disabled={props.busy}>⌫ {t.remove}</button>}
      <button data-hint={h.skip} type="button" onClick={props.onSkip}>{t.skip} →</button>
    </div>
  </section>;
}
