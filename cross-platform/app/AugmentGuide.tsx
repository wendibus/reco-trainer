import { useEffect, useState } from 'react';

type Language = 'de' | 'en' | 'es' | 'fr';

/** Shown each time the Augmentation Studio opens until "Don't show again" is ticked; the "Guide"
 *  button in the studio opens it again, and unticking the box there switches it back on.
 *  Screenshots: public/augment-guide/<language>-<step>.jpg, made by tests/augment-guide-shots.mjs
 *  from the real studio with drawn sample pictures (never real training images). */
export const GUIDE_HIDDEN_KEY = 'reco-augment-guide-hidden-v1';

export function guideHidden(): boolean {
  try { return window.localStorage.getItem(GUIDE_HIDDEN_KEY) === 'yes'; } catch { return false; }
}

function setGuideHidden(hidden: boolean) {
  try {
    if (hidden) window.localStorage.setItem(GUIDE_HIDDEN_KEY, 'yes');
    else window.localStorage.removeItem(GUIDE_HIDDEN_KEY);
  } catch { /* private mode: the guide simply shows again next time */ }
}

type Text = { title: string; step: string; back: string; next: string; done: string; close: string; hide: string; steps: [string, string][] };

export const guideText: Record<Language, Text & { open: string; openHint: string }> = {
  de: {
    title: 'So arbeitest du mit dem Augmentation Studio', step: 'Schritt', back: 'Zurück', next: 'Weiter', done: 'Los geht’s', close: 'Schließen', hide: 'Nicht mehr anzeigen',
    open: 'Anleitung', openHint: 'Zeigt die Schritt-für-Schritt-Anleitung. Dort lässt sie sich auch wieder beim Start einschalten.',
    steps: [
      ['Was das Studio macht', 'Das Studio baut aus deinen gelabelten Full-Court-Bildern neue Trainingsbilder. Es schneidet Spieler, Schiedsrichter und Ball aus, setzt sie an andere Stellen derselben Halle, mit passender Größe und Perspektive, und berechnet die Labels neu. Du brauchst einen vorbereiteten Arbeitsordner, „ML einrichten“ und mindestens 300 geprüfte, gelabelte Bilder aus mehreren Aufnahmen, besser 500.'],
      ['Quellordner wählen', 'Exportiere zuerst deinen Datensatz mit „Datensatz exportieren“ im COCO-Format und wähle diesen Ordner unter „Quellordner wählen“. Prüfe dann die Rolle je Klasse: Spieler und Schiedsrichter werden versetzt, der Ball wandert mit, Körbe und Tore bleiben an ihrem Platz, „Ignorieren“ lässt eine Klasse weg.'],
      ['Einstellungen', '„Anzahl Variationen“ ist die Zahl der erzeugten Bilder. Für einen ersten Test nimm 1.500 bis 2.000, etwa drei- bis fünfmal so viele wie echte Bilder. Der Anteil Ausschnitte erzeugt gezoomte Bildteile, die Zoomwerte legen deren Breite fest. Die Höchstzahlen verhindern überfüllte Felder, „Mind. sichtbar“ verwirft stark verdeckte Personen. Den Rest kannst du für den Anfang so lassen.'],
      ['Analysieren und Hallen prüfen', '„Analysieren“ berechnet Masken, die leere Halle je Aufnahme (Session) und das Spielfeld. Prüfe je Session, ob die Einteilung in Training, Validierung und Test stimmt und ob das Spielfeld richtig auf der leeren Halle liegt. Wenn nicht, „Spielfeld anklicken“ und die Ecken der Reihe nach setzen. Für eine ehrliche Bewertung braucht es mindestens drei Sessions.'],
      ['Teams prüfen', 'Spieler werden nach Trikotfarbe in zwei Teams sortiert, damit nie drei Mannschaften auf einem Bild stehen. Orange umrandete Personen sind unsicher: Ein Klick wechselt das Team. Blasse Personen sind verdeckt oder angeschnitten und werden nicht versetzt. Danach „Änderungen übernehmen und neu analysieren“.'],
      ['Erzeugen', '„Datensatz erzeugen“ rechnet die Bilder und prüft sie danach automatisch. Der Bericht sollte „0 Verstöße“ zeigen. Sieh dir die Vorschau an: Stehen die Personen glaubhaft auf dem Boden, passen die Boxen? Ein Abbruch schadet nicht, mit gleichen Einstellungen geht es dort weiter.'],
      ['Trainieren und vergleichen', 'Stell „Beim Training verwenden“ auf „Echt + synthetisch“ und starte „Lokal trainieren“. Trainiere danach mit „Nur echt, gleiche Aufteilung“ ein zweites Modell. Validierung und Test sind in beiden Läufen dieselben echten Bilder. Unter „Modelle testen“ siehst du, ob die erzeugten Bilder bei deinem Material helfen.'],
    ],
  },
  en: {
    title: 'How to work with the Augmentation Studio', step: 'Step', back: 'Back', next: 'Next', done: 'Let’s go', close: 'Close', hide: 'Don’t show again',
    open: 'Guide', openHint: 'Shows the step-by-step guide. You can also switch it back on at start there.',
    steps: [
      ['What the studio does', 'The studio builds new training images from your labelled full-court images. It cuts out players, referees and the ball, puts them elsewhere in the same venue with matching size and perspective, and computes the labels again. You need a prepared working folder, “Set up ML” and at least 300 reviewed, labelled images from several recordings, ideally 500.'],
      ['Select the source folder', 'First export your dataset with “Export dataset” in COCO format, then pick that folder under “Select source folder”. Check the role per class: players and referees are moved, the ball travels with them, hoops and goals stay in place, “Ignore” leaves a class out.'],
      ['Settings', '“Number of variations” is how many images are generated. For a first test use 1,500 to 2,000, about three to five times the number of real images. The share of crops creates zoomed parts of the image; the zoom values set their width. The maximums prevent crowded courts, “Min. visible” drops heavily occluded people. Leave the rest as it is for a start.'],
      ['Analyze and check the venues', '“Analyze” computes masks, the empty venue per recording (session) and the court. For each session, check that the split into training, validation and test is right and that the court sits correctly on the empty venue. If not, click “Click the court” and set the corners in order. An honest evaluation needs at least three sessions.'],
      ['Check the teams', 'Players are sorted into two teams by jersey colour, so an image never shows three teams. People outlined in orange are uncertain: a click changes the team. Pale people are occluded or cut off and are not moved. Then click “Apply changes and analyze again”.'],
      ['Generate', '“Generate dataset” renders the images and then checks them automatically. The report should say “0 violations”. Look at the preview: do people stand believably on the floor, do the boxes fit? Stopping does no harm; with the same settings it continues where it left off.'],
      ['Train and compare', 'Set “Use for training” to “Real + synthetic” and start “Train locally”. Then train a second model with “Real only, same split”. Validation and test are the same real images in both runs. “Test models” shows whether the generated images help with your footage.'],
    ],
  },
  es: {
    title: 'Cómo trabajar con Augmentation Studio', step: 'Paso', back: 'Atrás', next: 'Siguiente', done: 'Empezar', close: 'Cerrar', hide: 'No volver a mostrar',
    open: 'Guía', openHint: 'Muestra la guía paso a paso. Allí también puedes volver a activarla al inicio.',
    steps: [
      ['Qué hace el estudio', 'El estudio crea nuevas imágenes de entrenamiento a partir de tus imágenes etiquetadas de pista completa. Recorta jugadores, árbitros y balón, los coloca en otros puntos del mismo pabellón con el tamaño y la perspectiva adecuados y vuelve a calcular las etiquetas. Necesitas una carpeta de trabajo preparada, «Configurar ML» y al menos 300 imágenes revisadas y etiquetadas de varias grabaciones, mejor 500.'],
      ['Elegir la carpeta de origen', 'Primero exporta tu conjunto de datos con «Exportar conjunto de datos» en formato COCO y elige esa carpeta en «Seleccionar carpeta de origen». Revisa la función de cada clase: jugadores y árbitros se mueven, el balón los acompaña, canastas y porterías se quedan en su sitio, «Ignorar» deja fuera una clase.'],
      ['Ajustes', '«Número de variaciones» es la cantidad de imágenes generadas. Para una primera prueba usa entre 1.500 y 2.000, unas tres a cinco veces el número de imágenes reales. La parte de recortes crea partes ampliadas de la imagen; los valores de zoom fijan su anchura. Los máximos evitan pistas abarrotadas y «Mín. visible» descarta personas muy tapadas. El resto puede quedarse así para empezar.'],
      ['Analizar y revisar los pabellones', '«Analizar» calcula máscaras, el pabellón vacío de cada grabación (sesión) y la pista. En cada sesión, comprueba que el reparto en entrenamiento, validación y prueba sea correcto y que la pista esté bien colocada sobre el pabellón vacío. Si no, pulsa «Marcar el campo» y coloca las esquinas en orden. Una evaluación honesta necesita al menos tres sesiones.'],
      ['Revisar los equipos', 'Los jugadores se ordenan en dos equipos según el color de la camiseta, para que nunca aparezcan tres equipos en una imagen. Las personas con borde naranja son dudosas: un clic cambia el equipo. Las personas pálidas están tapadas o cortadas y no se mueven. Después pulsa «Aplicar cambios y volver a analizar».'],
      ['Generar', '«Generar conjunto de datos» crea las imágenes y luego las comprueba automáticamente. El informe debería indicar «0 infracciones». Mira la vista previa: ¿las personas pisan el suelo de forma creíble, encajan los cuadros? Detenerlo no hace daño; con los mismos ajustes continúa donde se quedó.'],
      ['Entrenar y comparar', 'Pon «Usar en el entrenamiento» en «Real + sintético» y pulsa «Entrenar localmente». Después entrena un segundo modelo con «Solo real, misma división». La validación y la prueba son las mismas imágenes reales en ambos casos. En «Probar modelos» verás si las imágenes generadas ayudan con tu material.'],
    ],
  },
  fr: {
    title: 'Travailler avec Augmentation Studio', step: 'Étape', back: 'Retour', next: 'Suivant', done: 'C’est parti', close: 'Fermer', hide: 'Ne plus afficher',
    open: 'Guide', openHint: 'Affiche le guide pas à pas. Vous pouvez aussi l’y réactiver au démarrage.',
    steps: [
      ['Ce que fait le studio', 'Le studio crée de nouvelles images d’entraînement à partir de vos images annotées du terrain entier. Il découpe joueurs, arbitres et ballon, les place ailleurs dans la même salle avec la bonne taille et la bonne perspective, puis recalcule les annotations. Il vous faut un dossier de travail préparé, « Configurer le ML » et au moins 300 images vérifiées et annotées issues de plusieurs enregistrements, idéalement 500.'],
      ['Choisir le dossier source', 'Exportez d’abord votre jeu de données avec « Exporter le jeu de données » au format COCO, puis choisissez ce dossier sous « Sélectionner le dossier source ». Vérifiez le rôle de chaque classe : joueurs et arbitres sont déplacés, le ballon les suit, paniers et buts restent à leur place, « Ignorer » écarte une classe.'],
      ['Réglages', '« Nombre de variations » est le nombre d’images générées. Pour un premier test, prenez 1 500 à 2 000, soit environ trois à cinq fois le nombre d’images réelles. La part de recadrages crée des parties zoomées de l’image ; les valeurs de zoom fixent leur largeur. Les maximums évitent les terrains surchargés, « Min. visible » écarte les personnes très masquées. Le reste peut rester tel quel pour commencer.'],
      ['Analyser et vérifier les salles', '« Analyser » calcule les masques, la salle vide de chaque enregistrement (session) et le terrain. Pour chaque session, vérifiez la répartition entre entraînement, validation et test, et que le terrain est bien placé sur la salle vide. Sinon, cliquez sur « Cliquer le terrain » et placez les coins dans l’ordre. Une évaluation honnête demande au moins trois sessions.'],
      ['Vérifier les équipes', 'Les joueurs sont répartis en deux équipes selon la couleur du maillot, pour qu’une image ne montre jamais trois équipes. Les personnes entourées d’orange sont incertaines : un clic change l’équipe. Les personnes pâles sont masquées ou coupées et ne sont pas déplacées. Ensuite, « Appliquer et analyser à nouveau ».'],
      ['Générer', '« Générer le jeu de données » crée les images puis les vérifie automatiquement. Le rapport doit indiquer « 0 violation ». Regardez l’aperçu : les personnes se tiennent-elles de façon crédible au sol, les boîtes sont-elles justes ? Une interruption ne pose pas de problème ; avec les mêmes réglages, la génération reprend où elle s’est arrêtée.'],
      ['Entraîner et comparer', 'Réglez « Utiliser pour l’entraînement » sur « Réel + synthétique » et lancez « Entraîner localement ». Entraînez ensuite un second modèle avec « Réel seul, même répartition ». Validation et test sont les mêmes images réelles dans les deux cas. « Tester les modèles » montre si les images générées aident avec vos vidéos.'],
    ],
  },
};

export function AugmentGuide({ language, onClose }: { language: Language; onClose: () => void }) {
  const t = guideText[language];
  const [step, setStep] = useState(0);
  const [hidden, setHidden] = useState(guideHidden);
  const last = step === t.steps.length - 1;
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
      else if (event.key === 'ArrowRight') setStep((value) => Math.min(t.steps.length - 1, value + 1));
      else if (event.key === 'ArrowLeft') setStep((value) => Math.max(0, value - 1));
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onClose, t.steps.length]);
  const [title, body] = t.steps[step];
  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="augment-guide" role="dialog" aria-modal="true" aria-labelledby="augment-guide-title">
      <button data-hint={t.close} type="button" className="walkthrough-close" aria-label={t.close} onClick={onClose}>×</button>
      <div className="walkthrough-kicker">{t.step} {step + 1} / {t.steps.length}</div>
      <h2 id="augment-guide-title">{t.title}</h2>
      <div className="augment-guide-progress" aria-hidden="true">{t.steps.map((_, index) => <span key={index} className={index <= step ? 'active' : ''} />)}</div>
      <div className="augment-guide-body">
        <figure><img key={`${language}-${step}`} src={`augment-guide/${language}-${step + 1}.jpg`} alt="" /></figure>
        <div><h3>{title}</h3><p>{body}</p></div>
      </div>
      <footer>
        <label className="cuda-toggle"><input type="checkbox" checked={hidden} onChange={(event) => { setHidden(event.target.checked); setGuideHidden(event.target.checked); }} />{t.hide}</label>
        <button data-hint={t.back} type="button" className="secondary-button" onClick={() => setStep((value) => Math.max(0, value - 1))} disabled={step === 0}>{t.back}</button>
        <button data-hint={last ? t.done : t.next} type="button" className="primary-button" onClick={() => { if (last) onClose(); else setStep((value) => value + 1); }}>{last ? t.done : t.next}</button>
      </footer>
    </section>
  </div>;
}
