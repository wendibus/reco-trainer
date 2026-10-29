export type ExtraLanguage = 'de' | 'en' | 'es' | 'fr';

export type ExtraText = {
  refine: string; refineHelp: string;
  onnx: string; coreml: string; coremlMacOnly: string;
  fieldSet: string; fieldEdit: string; fieldHelp: string; fieldNudge: string; notNow: string;
  showFolder: string;
  exportDataset: string; exportDatasetHelp: string;
  anonTrain: string; anonTrainHint: string; anonTrainOff: string;
  importLabels: string; importLabelsCandidates: string; importLabelsHelp: string;
  importOnnx: string; importOnnxHelp: string; onnxName: string; onnxClasses: string; onnxClassesHint: (allowed: string) => string; onnxInputSize: string; onnxInputSizeHint: string; onnxLicense: string; onnxChooseFile: string;
  editorIntro: string;
  pointPrompt: (index: number) => string; crossing: string;
  moreHint: string; pointCount: (count: number) => string; removeLast: string; reset: string; frameMissing: string; cancel: string; save: string; close: string;
};

export const extraText: Record<ExtraLanguage, ExtraText> = {
  de: {
    refine: 'Boxen mit OpenCV verfeinern',
    refineHelp: 'Zieht Ball-, Spieler-, Schiedsrichter- und weitere unterstützte Boxen lokal nach, auch von Hand gezeichnete. Nur plausible, eng anliegende Anpassungen werden übernommen; die ursprünglichen Koordinaten bleiben je Box gespeichert.',
    onnx: 'CPU-Modell (ONNX)', coreml: 'Apple-Modell (Core ML)', coremlMacOnly: 'Der Core-ML-Export ist nur auf einem Mac möglich.',
    fieldSet: 'Spielfeld festlegen', fieldEdit: 'Spielfeld bearbeiten',
    fieldHelp: 'Markiere den Rand des Spielfelds mit mindestens vier Punkten, damit „Automatisch markieren“ nur Personen berücksichtigt, die mit den Füßen auf dem Feld stehen.',
    fieldNudge: 'Spielfeld markieren, damit automatisches Markieren nur Personen auf dem Feld berücksichtigt.', notNow: 'Nicht jetzt',
    showFolder: 'Trainingsordner anzeigen',
    exportDataset: 'Datensatz exportieren', exportDatasetHelp: 'Alle geprüften Bilder mit Labels als YOLO- und/oder COCO-Datensatz exportieren, auf Wunsch anonymisiert.',
    anonTrain: 'Anonymisierte Trainingsbilder', anonTrainHint: 'Dieser Anteil der Trainingsbilder wird mit weichgezeichneten Köpfen trainiert, wie im anonymisierten Export. So erkennt das Modell Personen auch auf solchen Bildern.', anonTrainOff: 'aus',
    importLabels: 'Labels importieren', importLabelsCandidates: 'Als Vorschläge',
    importLabelsHelp: 'Übernimmt Bilder mit YOLO-txt- oder COCO-Labels aus einem Ordner (z. B. aus CVAT oder Roboflow). Klassennamen kommen aus data.yaml, classes.txt oder der COCO-Datei; unbekannte Klassen werden übersprungen. „Als Vorschläge“ legt die Boxen ungeprüft in die Prüfwarteschlange.',
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
    refineHelp: "Locally tightens ball, player, referee, and other supported boxes, including hand-drawn ones. Only plausible, closely-matching adjustments are applied; each box's original coordinates remain stored.",
    onnx: 'CPU model (ONNX)', coreml: 'Apple model (Core ML)', coremlMacOnly: 'Core ML export is only possible on a Mac.',
    fieldSet: 'Set field boundaries', fieldEdit: 'Edit field boundaries',
    fieldHelp: 'Mark the edge of the field with at least four points so "Auto-label" only considers people whose feet are standing on the field.',
    fieldNudge: 'Mark the field boundaries so auto-label only considers people standing on the field.', notNow: 'Not now',
    showFolder: 'Show training folder',
    exportDataset: 'Export dataset', exportDatasetHelp: 'Export every reviewed image with its labels as a YOLO and/or COCO dataset, anonymized if you like.',
    anonTrain: 'Anonymized training images', anonTrainHint: 'This share of the training images is trained with blurred heads, as in the anonymized export, so the model also recognizes people on such images.', anonTrainOff: 'off',
    importLabels: 'Import labels', importLabelsCandidates: 'As suggestions',
    importLabelsHelp: 'Imports images with YOLO-txt or COCO labels from a folder (e.g. from CVAT or Roboflow). Class names come from data.yaml, classes.txt, or the COCO file; unknown classes are skipped. “As suggestions” puts the boxes unreviewed into the review queue.',
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
    refineHelp: 'Ajusta localmente cuadros de balón, jugador, árbitro y otras clases compatibles, incluidos los dibujados a mano. Solo se aplican ajustes plausibles y cercanos; las coordenadas originales de cada cuadro permanecen guardadas.',
    onnx: 'Modelo CPU (ONNX)', coreml: 'Modelo Apple (Core ML)', coremlMacOnly: 'La exportación a Core ML solo es posible en un Mac.',
    fieldSet: 'Definir el campo', fieldEdit: 'Editar el campo',
    fieldHelp: 'Marca el borde del campo con al menos cuatro puntos para que «Marcado automático» solo considere a personas con los pies sobre el campo.',
    fieldNudge: 'Marca los límites del campo para que el marcado automático solo considere a personas dentro del campo.', notNow: 'Ahora no',
    showFolder: 'Mostrar carpeta de entrenamiento',
    exportDataset: 'Exportar conjunto de datos', exportDatasetHelp: 'Exporta todas las imágenes revisadas con sus etiquetas como conjunto YOLO y/o COCO, anonimizado si quieres.',
    anonTrain: 'Imágenes de entrenamiento anonimizadas', anonTrainHint: 'Esta parte de las imágenes de entrenamiento se entrena con cabezas difuminadas, como en la exportación anonimizada, para que el modelo también reconozca personas en esas imágenes.', anonTrainOff: 'no',
    importLabels: 'Importar etiquetas', importLabelsCandidates: 'Como sugerencias',
    importLabelsHelp: 'Importa imágenes con etiquetas YOLO-txt o COCO desde una carpeta (p. ej., de CVAT o Roboflow). Los nombres de clase se toman de data.yaml, classes.txt o del archivo COCO; las clases desconocidas se omiten. «Como sugerencias» coloca los cuadros sin revisar en la cola de revisión.',
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
    refineHelp: 'Resserre localement les boîtes ballon, joueur, arbitre et autres classes prises en charge, y compris dessinées à la main. Seuls les ajustements plausibles et proches sont appliqués ; les coordonnées d’origine de chaque boîte restent enregistrées.',
    onnx: 'Modèle CPU (ONNX)', coreml: 'Modèle Apple (Core ML)', coremlMacOnly: 'L’export Core ML n’est possible que sur un Mac.',
    fieldSet: 'Définir le terrain', fieldEdit: 'Modifier le terrain',
    fieldHelp: 'Marquez le bord du terrain avec au moins quatre points pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain.',
    fieldNudge: 'Marquez les limites du terrain pour que le marquage automatique ne prenne en compte que les personnes sur le terrain.', notNow: 'Pas maintenant',
    showFolder: 'Afficher le dossier d’entraînement',
    exportDataset: 'Exporter le jeu de données', exportDatasetHelp: 'Exporte toutes les images vérifiées avec leurs annotations en jeu YOLO et/ou COCO, anonymisé si souhaité.',
    anonTrain: 'Images d’entraînement anonymisées', anonTrainHint: 'Cette part des images d’entraînement est entraînée avec des têtes floutées, comme dans l’export anonymisé, pour que le modèle reconnaisse aussi les personnes sur de telles images.', anonTrainOff: 'non',
    importLabels: 'Importer des annotations', importLabelsCandidates: 'Comme suggestions',
    importLabelsHelp: 'Importe des images avec annotations YOLO-txt ou COCO depuis un dossier (p. ex. CVAT ou Roboflow). Les noms de classe proviennent de data.yaml, classes.txt ou du fichier COCO ; les classes inconnues sont ignorées. « Comme suggestions » place les boîtes non vérifiées dans la file de vérification.',
    importOnnx: 'Importer un modèle ONNX', onnxName: 'Nom', onnxClasses: 'Classes (ordre des ID)',
    importOnnxHelp: 'Pour les modèles YOLO exportés avec NMS intégré (sortie 1 × N × 6, p. ex. Ultralytics « end2end »). Le modèle sert au marquage automatique et à la comparaison ; il ne peut pas être entraîné.',
    onnxClassesHint: (allowed) => `Autorisées : ${allowed}. Les noms courants comme « person » ou « sports ball » sont convertis.`,
    onnxInputSize: 'Taille d’entrée (px)', onnxInputSizeHint: 'Nécessaire uniquement si le modèle n’a pas d’entrée carrée fixe.', onnxLicense: 'Licence (facultatif)', onnxChooseFile: 'Choisir le fichier ONNX et importer',
    editorIntro: 'Pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain. Cliquez au moins quatre points dans l’ordre le long du bord du terrain ; des lignes droites les relient. Si le terrain n’est pas entièrement dans l’image, placez des points sur le bord de l’image là où le terrain sort du cadre : toute forme est possible (jusqu’à 32 points). Les points peuvent être déplacés et le dernier peut être retiré.',
    pointPrompt: (index) => `Placez le point ${index} (au moins 4)`, crossing: 'Les bords se croisent : placez les points dans l’ordre le long du bord.',
    moreHint: "Cliquez pour ajouter des points, faites glisser un point pour le déplacer", pointCount: (count) => "Points : {n}".replace('{n}', String(count)), removeLast: "Supprimer le dernier point", reset: 'Réinitialiser', frameMissing: 'Image manquante', cancel: 'Annuler', save: 'Enregistrer', close: 'Fermer',
  },
};
