export type ExtraLanguage = 'de' | 'en' | 'es' | 'fr';

export type ExtraText = {
  refine: string;
  onnx: string; coreml: string;
  fieldSet: string; fieldEdit: string; fieldNudge: string; notNow: string;
  showFolder: string;
  exportDataset: string;
  anonTrain: string; anonTrainHint: string; anonTrainOff: string;
  consistency: string; flagDetector: string; flagVanished: string; flagVlm: string;
  importLabels: string; importLabelsCandidates: string;
  importOnnx: string; importOnnxHelp: string; onnxName: string; onnxClasses: string; onnxClassesHint: (allowed: string) => string; onnxInputSize: string; onnxInputSizeHint: string; onnxLicense: string; onnxChooseFile: string;
  editorIntro: string;
  pointPrompt: (index: number) => string; crossing: string;
  moreHint: string; pointCount: (count: number) => string; removeLast: string; reset: string; frameMissing: string; cancel: string; save: string; close: string;
};

export const extraText: Record<ExtraLanguage, ExtraText> = {
  de: {
    refine: 'Boxen mit OpenCV verfeinern',
    onnx: 'CPU-Modell (ONNX)', coreml: 'Apple-Modell (Core ML)',
    fieldSet: 'Spielfeld festlegen', fieldEdit: 'Spielfeld bearbeiten',
    fieldNudge: 'Spielfeld markieren, damit automatisches Markieren nur Personen auf dem Feld berücksichtigt.', notNow: 'Nicht jetzt',
    showFolder: 'Trainingsordner anzeigen',
    exportDataset: 'Datensatz exportieren',
    anonTrain: 'Anonymisierte Trainingsbilder', anonTrainHint: 'Dieser Anteil der Trainingsbilder wird mit weichgezeichneten Köpfen trainiert, wie im anonymisierten Export. So erkennt das Modell Personen auch auf solchen Bildern.', anonTrainOff: 'aus',
    consistency: 'Konsistenz prüfen', flagDetector: 'Zweiter Detektor uneinig', flagVanished: 'Spieler fehlen plötzlich', flagVlm: 'VLM: Boxen prüfen',
    importLabels: 'Labels importieren', importLabelsCandidates: 'Als Vorschläge',
    importOnnx: 'ONNX-Modell importieren', onnxName: 'Name', onnxClasses: 'Klassen (Reihenfolge der Klassen-IDs)',
    importOnnxHelp: 'Für YOLO-Modelle, die mit eingebautem NMS exportiert wurden (Ausgabe 1 × N × 6, z. B. Ultralytics „end2end“). Das Modell dient zum automatischen Markieren und zum Modellvergleich; trainieren lässt es sich nicht.',
    onnxClassesHint: (allowed) => `Erlaubt: ${allowed}. Übliche Namen wie „person“ oder „sports ball“ werden zugeordnet.`,
    onnxInputSize: 'Eingabegröße (px)', onnxInputSizeHint: 'Nur nötig, wenn das Modell keine feste quadratische Eingabe hat.', onnxLicense: 'Lizenz (optional)', onnxChooseFile: 'ONNX-Datei wählen und importieren',
    editorIntro: 'Damit „Automatisch markieren“ nur Personen berücksichtigt, die mit den Füßen auf dem Spielfeld stehen. Klicke mindestens vier Punkte der Reihe nach entlang des Spielfeldrands an; zwischen den Punkten gelten gerade Linien. Ist das Feld nicht ganz im Bild, setze die Punkte dort auf den Bildrand, wo das Feld das Bild verlässt – so sind beliebige Formen möglich (bis zu 32 Punkte). Punkte lassen sich verschieben, der letzte Punkt kann wieder entfernt werden.',
    pointPrompt: (index) => `Punkt ${index} setzen (mindestens 4)`, crossing: 'Die Kanten kreuzen sich – bitte die Punkte der Reihe nach am Rand entlang setzen.',
    moreHint: "Weitere Punkte klicken, Punkte ziehen zum Verschieben", pointCount: (count) => "Punkte: {n}".replace('{n}', String(count)), removeLast: "Letzten Punkt entfernen", reset: 'Zurücksetzen', frameMissing: 'Frame fehlt', cancel: 'Abbrechen', save: 'Speichern', close: 'Schließen',
  },
  en: {
    refine: 'Refine boxes with OpenCV',
    onnx: 'CPU model (ONNX)', coreml: 'Apple model (Core ML)',
    fieldSet: 'Set field boundaries', fieldEdit: 'Edit field boundaries',
    fieldNudge: 'Mark the field boundaries so auto-label only considers people standing on the field.', notNow: 'Not now',
    showFolder: 'Show training folder',
    exportDataset: 'Export dataset',
    anonTrain: 'Anonymized training images', anonTrainHint: 'This share of the training images is trained with blurred heads, as in the anonymized export, so the model also recognizes people on such images.', anonTrainOff: 'off',
    consistency: 'Check consistency', flagDetector: 'Second detector disagrees', flagVanished: 'Players suddenly missing', flagVlm: 'VLM: check boxes',
    importLabels: 'Import labels', importLabelsCandidates: 'As suggestions',
    importOnnx: 'Import ONNX model', onnxName: 'Name', onnxClasses: 'Classes (class-id order)',
    importOnnxHelp: 'For YOLO models exported with NMS built in (output 1 × N × 6, e.g. Ultralytics “end2end”). The model is used for auto-labeling and the model comparison; it cannot be trained.',
    onnxClassesHint: (allowed) => `Allowed: ${allowed}. Common names such as “person” or “sports ball” are mapped.`,
    onnxInputSize: 'Input size (px)', onnxInputSizeHint: 'Only needed when the model has no fixed square input.', onnxLicense: 'License (optional)', onnxChooseFile: 'Choose ONNX file and import',
    editorIntro: 'So "Auto-label" only considers people whose feet are standing on the field. Click at least four points in order along the edge of the field; straight lines join them. If the field is not fully in the image, put points on the image edge where the field leaves the frame - any shape is possible (up to 32 points). Points can be dragged, and the last point can be removed again.',
    pointPrompt: (index) => `Place point ${index} (at least 4)`, crossing: 'The edges cross - please place the points in order along the edge.',
    moreHint: "Click to add more points, drag a point to move it", pointCount: (count) => "Points: {n}".replace('{n}', String(count)), removeLast: "Remove last point", reset: 'Reset', frameMissing: 'Frame is missing', cancel: 'Cancel', save: 'Save', close: 'Close',
  },
  es: {
    refine: 'Refinar cuadros con OpenCV',
    onnx: 'Modelo CPU (ONNX)', coreml: 'Modelo Apple (Core ML)',
    fieldSet: 'Definir el campo', fieldEdit: 'Editar el campo',
    fieldNudge: 'Marca los límites del campo para que el marcado automático solo considere a personas dentro del campo.', notNow: 'Ahora no',
    showFolder: 'Mostrar carpeta de entrenamiento',
    exportDataset: 'Exportar conjunto de datos',
    anonTrain: 'Imágenes de entrenamiento anonimizadas', anonTrainHint: 'Esta parte de las imágenes de entrenamiento se entrena con cabezas difuminadas, como en la exportación anonimizada, para que el modelo también reconozca personas en esas imágenes.', anonTrainOff: 'no',
    consistency: 'Comprobar coherencia', flagDetector: 'El segundo detector discrepa', flagVanished: 'Faltan jugadores de repente', flagVlm: 'VLM: revisar cuadros',
    importLabels: 'Importar etiquetas', importLabelsCandidates: 'Como sugerencias',
    importOnnx: 'Importar modelo ONNX', onnxName: 'Nombre', onnxClasses: 'Clases (orden de ID)',
    importOnnxHelp: 'Para modelos YOLO exportados con NMS integrado (salida 1 × N × 6, p. ej. Ultralytics «end2end»). El modelo sirve para el marcado automático y la comparación; no se puede entrenar.',
    onnxClassesHint: (allowed) => `Permitidas: ${allowed}. Se asignan nombres habituales como «person» o «sports ball».`,
    onnxInputSize: 'Tamaño de entrada (px)', onnxInputSizeHint: 'Solo es necesario si el modelo no tiene una entrada cuadrada fija.', onnxLicense: 'Licencia (opcional)', onnxChooseFile: 'Elegir archivo ONNX e importar',
    editorIntro: 'Para que «Marcado automático» solo considere a personas con los pies sobre el campo. Haz clic en al menos cuatro puntos en orden a lo largo del borde del campo; se unen con líneas rectas. Si el campo no está completo en la imagen, coloca puntos en el borde de la imagen donde el campo sale del encuadre: cualquier forma es posible (hasta 32 puntos). Los puntos se pueden arrastrar y el último se puede quitar.',
    pointPrompt: (index) => `Coloca el punto ${index} (al menos 4)`, crossing: 'Los bordes se cruzan: coloca los puntos en orden a lo largo del borde.',
    moreHint: "Haz clic para añadir más puntos; arrastra un punto para moverlo", pointCount: (count) => "Puntos: {n}".replace('{n}', String(count)), removeLast: "Quitar el último punto", reset: 'Reiniciar', frameMissing: 'Falta el fotograma', cancel: 'Cancelar', save: 'Guardar', close: 'Cerrar',
  },
  fr: {
    refine: 'Affiner les boîtes avec OpenCV',
    onnx: 'Modèle CPU (ONNX)', coreml: 'Modèle Apple (Core ML)',
    fieldSet: 'Définir le terrain', fieldEdit: 'Modifier le terrain',
    fieldNudge: 'Marquez les limites du terrain pour que le marquage automatique ne prenne en compte que les personnes sur le terrain.', notNow: 'Pas maintenant',
    showFolder: 'Afficher le dossier d’entraînement',
    exportDataset: 'Exporter le jeu de données',
    anonTrain: 'Images d’entraînement anonymisées', anonTrainHint: 'Cette part des images d’entraînement est entraînée avec des têtes floutées, comme dans l’export anonymisé, pour que le modèle reconnaisse aussi les personnes sur de telles images.', anonTrainOff: 'non',
    consistency: 'Vérifier la cohérence', flagDetector: 'Second détecteur en désaccord', flagVanished: 'Joueurs soudainement absents', flagVlm: 'VLM : vérifier les boîtes',
    importLabels: 'Importer des annotations', importLabelsCandidates: 'Comme suggestions',
    importOnnx: 'Importer un modèle ONNX', onnxName: 'Nom', onnxClasses: 'Classes (ordre des ID)',
    importOnnxHelp: 'Pour les modèles YOLO exportés avec NMS intégré (sortie 1 × N × 6, p. ex. Ultralytics « end2end »). Le modèle sert au marquage automatique et à la comparaison ; il ne peut pas être entraîné.',
    onnxClassesHint: (allowed) => `Autorisées : ${allowed}. Les noms courants comme « person » ou « sports ball » sont convertis.`,
    onnxInputSize: 'Taille d’entrée (px)', onnxInputSizeHint: 'Nécessaire uniquement si le modèle n’a pas d’entrée carrée fixe.', onnxLicense: 'Licence (facultatif)', onnxChooseFile: 'Choisir le fichier ONNX et importer',
    editorIntro: 'Pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain. Cliquez au moins quatre points dans l’ordre le long du bord du terrain ; des lignes droites les relient. Si le terrain n’est pas entièrement dans l’image, placez des points sur le bord de l’image là où le terrain sort du cadre : toute forme est possible (jusqu’à 32 points). Les points peuvent être déplacés et le dernier peut être retiré.',
    pointPrompt: (index) => `Placez le point ${index} (au moins 4)`, crossing: 'Les bords se croisent : placez les points dans l’ordre le long du bord.',
    moreHint: "Cliquez pour ajouter des points, faites glisser un point pour le déplacer", pointCount: (count) => "Points : {n}".replace('{n}', String(count)), removeLast: "Supprimer le dernier point", reset: 'Réinitialiser', frameMissing: 'Image manquante', cancel: 'Annuler', save: 'Enregistrer', close: 'Fermer',
  },
};
