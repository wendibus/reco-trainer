import { useState } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';
type Package = { packageID: string; displayName?: string; classes: string[]; isActive?: boolean };

const TEXT: Record<Language, { button: string; buttonHint: string; title: string; intro: string; model: string; active: string; classes: string; classesHint: string; ballOnly: string; noModels: string; cancel: string; cancelHint: string; run: string; runHint: string }> = {
  de: {
    button: 'Für Reco exportieren',
    buttonHint: 'Speichert ein Modell aus der Bibliothek als ONNX-Datei, die die Reco-Kamera-App direkt laden kann (Feld „AI model (ONNX)“ oder --model). Auch kombinierte Modelle.',
    title: 'Für Reco exportieren',
    intro: 'Reco liest Modelle im YOLO-Format: ein Eingang mit grauen Rändern, eine Ausgabe mit Box, Sicherheit und Klasse, die Klassennamen in der Datei. Reco Trainer baut die Datei so und prüft sie danach auf Bildern dieses Projekts. Die Datei landet im Projekt unter exports/reco.',
    model: 'Modell',
    active: 'aktiv',
    classes: 'Klassen in der Datei',
    classesHint: 'Reco verfolgt den Ball und findet Spieler unter dem Namen „person“; Spieler werden dafür umbenannt. Nicht gewählte Klassen fehlen in der Datei.',
    ballOnly: 'Nur Ball: in Reco den Tracking-Modus „ball“ wählen.',
    noModels: 'Noch kein Modell in der Bibliothek. Zuerst trainieren oder ein Modell laden.',
    cancel: 'Abbrechen', cancelHint: 'Schließt das Fenster ohne Export.',
    run: 'Exportieren', runHint: 'Erstellt die ONNX-Datei für Reco und prüft sie.',
  },
  en: {
    button: 'Export for Reco',
    buttonHint: 'Saves a library model as an ONNX file the Reco camera app loads directly ("AI model (ONNX)" field or --model). Combined models too.',
    title: 'Export for Reco',
    intro: 'Reco reads models in the YOLO layout: one input with grey borders, one output with box, confidence and class, and the class names inside the file. Reco Trainer builds the file that way and then checks it on images of this project. The file goes into the project under exports/reco.',
    model: 'Model',
    active: 'active',
    classes: 'Classes in the file',
    classesHint: 'Reco tracks the ball and finds players under the name "person", so players are renamed. Classes you leave out are not in the file.',
    ballOnly: 'Ball only: choose the tracking mode "ball" in Reco.',
    noModels: 'No model in the library yet. Train or load a model first.',
    cancel: 'Cancel', cancelHint: 'Closes this window without exporting.',
    run: 'Export', runHint: 'Creates the ONNX file for Reco and checks it.',
  },
  es: {
    button: 'Exportar para Reco',
    buttonHint: 'Guarda un modelo de la biblioteca como archivo ONNX que la app de cámara Reco carga directamente (campo «AI model (ONNX)» o --model). También modelos combinados.',
    title: 'Exportar para Reco',
    intro: 'Reco lee modelos en formato YOLO: una entrada con bordes grises, una salida con caja, confianza y clase, y los nombres de clase dentro del archivo. Reco Trainer crea el archivo así y luego lo comprueba con imágenes de este proyecto. El archivo queda en el proyecto en exports/reco.',
    model: 'Modelo',
    active: 'activo',
    classes: 'Clases en el archivo',
    classesHint: 'Reco sigue el balón y encuentra jugadores con el nombre «person», por eso se renombran. Las clases no elegidas no están en el archivo.',
    ballOnly: 'Solo balón: en Reco elige el modo de seguimiento «ball».',
    noModels: 'Aún no hay modelos en la biblioteca. Primero entrena o carga un modelo.',
    cancel: 'Cancelar', cancelHint: 'Cierra esta ventana sin exportar.',
    run: 'Exportar', runHint: 'Crea el archivo ONNX para Reco y lo comprueba.',
  },
  fr: {
    button: 'Exporter pour Reco',
    buttonHint: 'Enregistre un modèle de la bibliothèque en fichier ONNX que l’app caméra Reco charge directement (champ « AI model (ONNX) » ou --model). Aussi les modèles combinés.',
    title: 'Exporter pour Reco',
    intro: 'Reco lit les modèles au format YOLO : une entrée avec bords gris, une sortie avec boîte, confiance et classe, et les noms de classes dans le fichier. Reco Trainer construit le fichier ainsi puis le vérifie sur des images de ce projet. Le fichier va dans le projet sous exports/reco.',
    model: 'Modèle',
    active: 'actif',
    classes: 'Classes dans le fichier',
    classesHint: 'Reco suit le ballon et trouve les joueurs sous le nom « person » : les joueurs sont renommés. Les classes non choisies ne sont pas dans le fichier.',
    ballOnly: 'Ballon seul : dans Reco, choisissez le mode de suivi « ball ».',
    noModels: 'Aucun modèle dans la bibliothèque. Entraînez ou chargez d’abord un modèle.',
    cancel: 'Annuler', cancelHint: 'Ferme cette fenêtre sans exporter.',
    run: 'Exporter', runHint: 'Crée le fichier ONNX pour Reco et le vérifie.',
  },
};

export const recoExportText = TEXT;

/** Ball is preselected; without a ball class, the model's first class. */
export function defaultRecoClasses(classes: string[]): string[] {
  return classes.includes('ball') ? ['ball'] : classes.slice(0, 1);
}

export default function RecoExportDialog({ language, packages, categoryName, onExport, onCancel }: { language: Language; packages: Package[]; categoryName: (value: string) => string; onExport: (packageID: string, classes: string[]) => void; onCancel: () => void }) {
  const t = TEXT[language];
  const first = packages.find((item) => item.isActive) ?? packages[0];
  const [packageID, setPackageID] = useState(first?.packageID ?? '');
  const selected = packages.find((item) => item.packageID === packageID);
  const [classes, setClasses] = useState<string[]>(defaultRecoClasses(first?.classes ?? []));
  const choose = (id: string) => {
    setPackageID(id);
    setClasses(defaultRecoClasses(packages.find((item) => item.packageID === id)?.classes ?? []));
  };
  const toggle = (value: string) => setClasses((current) => current.includes(value) ? current.filter((item) => item !== value) : (selected?.classes ?? []).filter((item) => item === value || current.includes(item)));
  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onCancel(); }}>
    <form className="onnx-dialog reco-export-dialog" role="dialog" aria-modal="true" aria-labelledby="reco-export-title" onSubmit={(event) => { event.preventDefault(); if (packageID && classes.length) onExport(packageID, classes); }}>
      <h2 id="reco-export-title">{t.title}</h2>
      <p>{t.intro}</p>
      {!packages.length && <p className="split-problem">{t.noModels}</p>}
      {!!packages.length && <label>{t.model}<select value={packageID} onChange={(event) => choose(event.target.value)}>{packages.map((item) => <option key={item.packageID} value={item.packageID}>{(item.displayName || item.packageID) + (item.isActive ? ` (${t.active})` : '')}</option>)}</select></label>}
      {selected && <fieldset className="reco-export-classes"><legend>{t.classes}</legend>{selected.classes.map((item) => <label key={item}><input type="checkbox" checked={classes.includes(item)} onChange={() => toggle(item)} />{categoryName(item)}{item === 'player' && <small> → person</small>}</label>)}<small>{t.classesHint}</small></fieldset>}
      {classes.length === 1 && classes[0] === 'ball' && <small>{t.ballOnly}</small>}
      <div className="onnx-actions">
        <button data-hint={t.cancelHint} type="button" className="secondary-button" onClick={onCancel}>{t.cancel}</button>
        <button data-hint={t.runHint} type="submit" className="primary-button" disabled={!packageID || !classes.length}>{t.run}</button>
      </div>
    </form>
  </div>;
}
