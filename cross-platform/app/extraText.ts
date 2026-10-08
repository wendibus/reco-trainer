export type ExtraLanguage = 'de' | 'en' | 'es' | 'fr';

export type ExtraText = {
  refine: string; refineHelp: string;
  onnx: string; coreml: string; coremlMacOnly: string;
  fieldSet: string; fieldEdit: string; fieldHelp: string; fieldNudge: string; notNow: string;
  showFolder: string;
  editorIntro: string; widthLabel: string; lengthLabel: string;
  cornerLabels: [string, string, string, string];
  cornerPrompt: (index: number, label: string) => string;
  allSet: string; moreHint: string; pointCount: (count: number) => string; removeLast: string; sizeOptional: string; reset: string; frameMissing: string; cancel: string; save: string; close: string;
};

export const extraText: Record<ExtraLanguage, ExtraText> = {
  de: {
    refine: 'Boxen mit OpenCV verfeinern',
    refineHelp: 'Zieht Ball-, Spieler-, Schiedsrichter- und weitere unterstützte Boxen lokal nach, auch von Hand gezeichnete. Nur plausible, eng anliegende Anpassungen werden übernommen; die ursprünglichen Koordinaten bleiben je Box gespeichert.',
    onnx: 'CPU-Modell (ONNX)', coreml: 'Apple-Modell (Core ML)', coremlMacOnly: 'Der Core-ML-Export ist nur auf einem Mac möglich.',
    fieldSet: 'Spielfeld festlegen', fieldEdit: 'Spielfeld bearbeiten',
    fieldHelp: 'Markiere die Eckpunkte des Spielfelds (vier Ecken oder mehr Punkte entlang des Randes), damit „Automatisch markieren“ nur Personen berücksichtigt, die mit den Füßen auf dem Feld stehen.',
    fieldNudge: 'Spielfeld markieren, damit automatisches Markieren nur Personen auf dem Feld berücksichtigt.', notNow: 'Nicht jetzt',
    showFolder: 'Trainingsordner anzeigen',
    editorIntro: "Damit „Automatisch markieren“ nur Personen berücksichtigt, die mit den Füßen auf dem Spielfeld stehen. Klicke die Eckpunkte des Spielfelds im Bild an - bei einem rechteckigen Feld die vier Ecken in dieser Reihenfolge: oben links, oben rechts, unten rechts, unten links - und gib die echten Feldmaße ein. Bei anderen Formen setzt du einfach weitere Punkte rund um den Rand (bis zu 32); die Maße sind dann optional. Punkte lassen sich verschieben, der letzte Punkt kann wieder entfernt werden.",
    widthLabel: 'Breite (m)', lengthLabel: 'Länge (m)',
    cornerLabels: ['Oben links', 'Oben rechts', 'Unten rechts', 'Unten links'],
    cornerPrompt: (index, label) => `Ecke ${index}/4: ${label}`,
    allSet: 'Alle 4 Ecken gesetzt', moreHint: "Weitere Punkte klicken, Punkte ziehen zum Verschieben", pointCount: (count) => "Punkte: {n}".replace('{n}', String(count)), removeLast: "Letzten Punkt entfernen", sizeOptional: " (bei mehr als 4 Punkten optional)", reset: 'Zurücksetzen', frameMissing: 'Frame fehlt', cancel: 'Abbrechen', save: 'Speichern', close: 'Schließen',
  },
  en: {
    refine: 'Refine boxes with OpenCV',
    refineHelp: "Locally tightens ball, player, referee, and other supported boxes, including hand-drawn ones. Only plausible, closely-matching adjustments are applied; each box's original coordinates remain stored.",
    onnx: 'CPU model (ONNX)', coreml: 'Apple model (Core ML)', coremlMacOnly: 'Core ML export is only possible on a Mac.',
    fieldSet: 'Set field boundaries', fieldEdit: 'Edit field boundaries',
    fieldHelp: 'Mark the field\'s corner points (four corners or more points along the edge) so "Auto-label" only considers people whose feet are standing on the field.',
    fieldNudge: 'Mark the field boundaries so auto-label only considers people standing on the field.', notNow: 'Not now',
    showFolder: 'Show training folder',
    editorIntro: "So \"Auto-label\" only considers people whose feet are standing on the field. Click the field's corner points in the image - for a rectangular field its four corners in this order: top left, top right, bottom right, bottom left - and enter the field's real dimensions. For other shapes just keep adding points around the outline (up to 32); the dimensions are then optional. Drag a point to move it, or remove the last point again.",
    widthLabel: 'Width (m)', lengthLabel: 'Length (m)',
    cornerLabels: ['Top left', 'Top right', 'Bottom right', 'Bottom left'],
    cornerPrompt: (index, label) => `Corner ${index}/4: ${label}`,
    allSet: 'All 4 corners set', moreHint: "Click to add more points, drag a point to move it", pointCount: (count) => "Points: {n}".replace('{n}', String(count)), removeLast: "Remove last point", sizeOptional: " (optional with more than 4 points)", reset: 'Reset', frameMissing: 'Frame is missing', cancel: 'Cancel', save: 'Save', close: 'Close',
  },
  es: {
    refine: 'Refinar cuadros con OpenCV',
    refineHelp: 'Ajusta localmente cuadros de balón, jugador, árbitro y otras clases compatibles, incluidos los dibujados a mano. Solo se aplican ajustes plausibles y cercanos; las coordenadas originales de cada cuadro permanecen guardadas.',
    onnx: 'Modelo CPU (ONNX)', coreml: 'Modelo Apple (Core ML)', coremlMacOnly: 'La exportación a Core ML solo es posible en un Mac.',
    fieldSet: 'Definir el campo', fieldEdit: 'Editar el campo',
    fieldHelp: 'Marca los puntos del campo (cuatro esquinas o más puntos a lo largo del borde) para que «Marcado automático» solo considere a personas con los pies sobre el campo.',
    fieldNudge: 'Marca los límites del campo para que el marcado automático solo considere a personas dentro del campo.', notNow: 'Ahora no',
    showFolder: 'Mostrar carpeta de entrenamiento',
    editorIntro: "Para que «Marcado automático» solo considere a personas con los pies sobre el campo. Haz clic en los puntos del contorno del campo en la imagen: en un campo rectangular, sus cuatro esquinas en este orden: superior izquierda, superior derecha, inferior derecha, inferior izquierda, e introduce las medidas reales. Para otras formas sigue añadiendo puntos alrededor del borde (hasta 32); las medidas son entonces opcionales. Arrastra un punto para moverlo o quita el último.",
    widthLabel: 'Ancho (m)', lengthLabel: 'Largo (m)',
    cornerLabels: ['Superior izquierda', 'Superior derecha', 'Inferior derecha', 'Inferior izquierda'],
    cornerPrompt: (index, label) => `Esquina ${index}/4: ${label}`,
    allSet: 'Las 4 esquinas listas', moreHint: "Haz clic para añadir más puntos; arrastra un punto para moverlo", pointCount: (count) => "Puntos: {n}".replace('{n}', String(count)), removeLast: "Quitar el último punto", sizeOptional: " (opcional con más de 4 puntos)", reset: 'Reiniciar', frameMissing: 'Falta el fotograma', cancel: 'Cancelar', save: 'Guardar', close: 'Cerrar',
  },
  fr: {
    refine: 'Affiner les boîtes avec OpenCV',
    refineHelp: 'Resserre localement les boîtes ballon, joueur, arbitre et autres classes prises en charge, y compris dessinées à la main. Seuls les ajustements plausibles et proches sont appliqués ; les coordonnées d’origine de chaque boîte restent enregistrées.',
    onnx: 'Modèle CPU (ONNX)', coreml: 'Modèle Apple (Core ML)', coremlMacOnly: 'L’export Core ML n’est possible que sur un Mac.',
    fieldSet: 'Définir le terrain', fieldEdit: 'Modifier le terrain',
    fieldHelp: 'Marquez les points du terrain (quatre coins ou davantage de points le long du bord) pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain.',
    fieldNudge: 'Marquez les limites du terrain pour que le marquage automatique ne prenne en compte que les personnes sur le terrain.', notNow: 'Pas maintenant',
    showFolder: 'Afficher le dossier d’entraînement',
    editorIntro: "Pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain. Cliquez sur les points du contour du terrain dans l’image - pour un terrain rectangulaire, ses quatre coins dans l’ordre : en haut à gauche, en haut à droite, en bas à droite, en bas à gauche - et saisissez les dimensions réelles. Pour d’autres formes, continuez à ajouter des points le long du bord (jusqu’à 32) ; les dimensions sont alors facultatives. Faites glisser un point pour le déplacer ou supprimez le dernier.",
    widthLabel: 'Largeur (m)', lengthLabel: 'Longueur (m)',
    cornerLabels: ['En haut à gauche', 'En haut à droite', 'En bas à droite', 'En bas à gauche'],
    cornerPrompt: (index, label) => `Coin ${index}/4 : ${label}`,
    allSet: 'Les 4 coins sont placés', moreHint: "Cliquez pour ajouter des points, faites glisser un point pour le déplacer", pointCount: (count) => "Points : {n}".replace('{n}', String(count)), removeLast: "Supprimer le dernier point", sizeOptional: " (facultatif au-delà de 4 points)", reset: 'Réinitialiser', frameMissing: 'Image manquante', cancel: 'Annuler', save: 'Enregistrer', close: 'Fermer',
  },
};
