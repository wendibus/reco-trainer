type Language = 'de' | 'en' | 'es' | 'fr';

const TEXT: Record<Language, { title: string; body: string; set: string; setHint: string; skip: string; skipHint: string; cancel: string; cancelHint: string }> = {
  de: {
    title: 'Erst das Spielfeld festlegen?',
    body: 'Mit einem Spielfeld markiert das Vormarkieren nur Personen, die auf dem Feld stehen, keine Zuschauer, Bank oder Betreuer. Lege es jetzt fest und starte danach noch einmal. Mit „Für alle Videos übernehmen“ gilt es auch für neue Videos.',
    set: 'Spielfeld festlegen', setHint: 'Öffnet den Spielfeld-Editor. Das Vormarkieren startest du danach noch einmal.',
    skip: 'Ohne Spielfeld fortfahren', skipHint: 'Startet das Vormarkieren jetzt ohne Spielfeld. Diese Frage kommt in dieser Sitzung nicht wieder.',
    cancel: 'Abbrechen', cancelHint: 'Schließt das Fenster, ohne zu starten.',
  },
  en: {
    title: 'Set the field boundary first?',
    body: 'With a field boundary, pre-labelling only marks people standing on the field, not spectators, the bench or staff. Set it now and start again afterwards. With “Apply to all videos” it also covers new videos.',
    set: 'Set field boundaries', setHint: 'Opens the field editor. You start pre-labelling again afterwards.',
    skip: 'Continue without a field', skipHint: 'Starts pre-labelling now without a field. This question does not come back in this session.',
    cancel: 'Cancel', cancelHint: 'Closes the window without starting.',
  },
  es: {
    title: '¿Definir primero el campo?',
    body: 'Con un campo definido, el premarcado solo marca a las personas que están en el campo, no a espectadores, banquillo o personal. Defínelo ahora y vuelve a empezar después. Con «Aplicar a todos los vídeos» también vale para vídeos nuevos.',
    set: 'Definir el campo', setHint: 'Abre el editor de campo. Después vuelves a iniciar el premarcado.',
    skip: 'Continuar sin campo', skipHint: 'Inicia el premarcado ahora sin campo. Esta pregunta no vuelve en esta sesión.',
    cancel: 'Cancelar', cancelHint: 'Cierra la ventana sin iniciar.',
  },
  fr: {
    title: 'Définir d’abord le terrain ?',
    body: 'Avec un terrain défini, la pré-annotation ne marque que les personnes sur le terrain, pas les spectateurs, le banc ou le staff. Définissez-le maintenant et relancez ensuite. Avec « Appliquer à toutes les vidéos », il vaut aussi pour les nouvelles vidéos.',
    set: 'Définir le terrain', setHint: 'Ouvre l’éditeur de terrain. Vous relancez ensuite la pré-annotation.',
    skip: 'Continuer sans terrain', skipHint: 'Lance la pré-annotation maintenant sans terrain. Cette question ne revient pas pendant cette session.',
    cancel: 'Annuler', cancelHint: 'Ferme la fenêtre sans démarrer.',
  },
};

export const fieldOfferText = TEXT;

/** Asked before pre-labelling when no field boundary exists yet. */
export default function FieldOfferDialog({ language, onSetField, onContinue, onCancel }: { language: Language; onSetField: () => void; onContinue: () => void; onCancel: () => void }) {
  const t = TEXT[language];
  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
    <section className="onnx-dialog field-offer-dialog" role="alertdialog" aria-modal="true" aria-labelledby="field-offer-title">
      <h2 id="field-offer-title">{t.title}</h2>
      <p>{t.body}</p>
      <div className="onnx-actions">
        <button data-hint={t.cancelHint} type="button" className="secondary-button" onClick={onCancel}>{t.cancel}</button>
        <button data-hint={t.skipHint} type="button" className="secondary-button" onClick={onContinue}>{t.skip}</button>
        <button data-hint={t.setHint} type="button" className="primary-button" onClick={onSetField}>{t.set}</button>
      </div>
    </section>
  </div>;
}
