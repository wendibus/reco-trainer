import { useEffect } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type Item = { title: string; text: string };

/** Shown once after an update (see page.tsx, key reco-whats-new-seen-v1); clicking the version
 *  number in the header opens it again. Add the next version's entry at the top. */
export const WHATS_NEW: { version: string; items: Record<Language, Item[]> }[] = [
  {
    version: '0.31.5',
    items: {
      de: [{ title: 'Saubereres Protokoll beim Training auf der AMD-Karte', text: 'Beim Training auf der AMD-Karte stand am Anfang mehrfach „Unknown command line argument … offload-arch.exe“. Die Meldung kam von einer Hardware-Abfrage von AMD, die an Ordnernamen mit Leerzeichen scheitert. Sie war harmlos und entfällt jetzt; das Training selbst ändert sich nicht.' }],
      en: [{ title: 'Cleaner log when training on the AMD card', text: 'Training on the AMD card started with several “Unknown command line argument … offload-arch.exe” lines. They came from an AMD hardware query that fails on folder names with spaces. They were harmless and are gone now; training itself is unchanged.' }],
      es: [{ title: 'Registro más limpio al entrenar en la tarjeta AMD', text: 'El entrenamiento en la tarjeta AMD empezaba con varias líneas «Unknown command line argument … offload-arch.exe». Venían de una consulta de hardware de AMD que falla con nombres de carpeta con espacios. Eran inofensivas y ya no aparecen; el entrenamiento no cambia.' }],
      fr: [{ title: 'Journal plus propre lors de l’entraînement sur la carte AMD', text: 'L’entraînement sur la carte AMD commençait par plusieurs lignes « Unknown command line argument … offload-arch.exe ». Elles venaient d’une requête matérielle d’AMD qui échoue avec des noms de dossier contenant des espaces. Elles étaient sans conséquence et ont disparu ; l’entraînement lui-même ne change pas.' }],
    },
  },
  {
    version: '0.31.4',
    items: {
      de: [{ title: 'YOLO trainiert auf der AMD-Karte', text: 'Ist „AMD-Training einrichten“ erledigt und „Auf der AMD-Karte trainieren“ an, trainieren jetzt auch YOLO-Modelle auf der Karte. Beim ersten Batch-Training mit YOLO wird YOLO dafür zusätzlich in die AMD-Umgebung installiert (einige hundert MB). Klappt das nicht, trainiert YOLO wie bisher auf der CPU.' }],
      en: [{ title: 'YOLO trains on the AMD card', text: 'With “Set up AMD training” done and “Train on the AMD card” switched on, YOLO models now train on the card too. The first batch training with YOLO also installs YOLO into the AMD environment (a few hundred MB). If that fails, YOLO trains on the CPU as before.' }],
      es: [{ title: 'YOLO se entrena en la tarjeta AMD', text: 'Con «Configurar entrenamiento AMD» completado y «Entrenar en la tarjeta AMD» activado, los modelos YOLO ahora también se entrenan en la tarjeta. El primer entrenamiento por lotes con YOLO instala YOLO además en el entorno AMD (unos cientos de MB). Si falla, YOLO se entrena en la CPU como antes.' }],
      fr: [{ title: 'YOLO s’entraîne sur la carte AMD', text: 'Une fois « Configurer l’entraînement AMD » terminé et « Entraîner sur la carte AMD » activé, les modèles YOLO s’entraînent désormais aussi sur la carte. Le premier entraînement par lots avec YOLO installe en plus YOLO dans l’environnement AMD (quelques centaines de Mo). En cas d’échec, YOLO s’entraîne sur le CPU comme avant.' }],
    },
  },
  {
    version: '0.31.3',
    items: {
      de: [{ title: 'Batch-Training mit YOLO nutzt die AMD-Karte für RF-DETR', text: 'Enthielt ein Batch-Training auch ein YOLO-Modell, lief bisher der ganze Batch auf der CPU, obwohl „Auf der AMD-Karte trainieren“ eingeschaltet war. Jetzt trainieren die RF-DETR-Modelle des Batches auf der Karte, nur YOLO bleibt auf der CPU.' }],
      en: [{ title: 'Batch training with YOLO uses the AMD card for RF-DETR', text: 'When a batch training included a YOLO model, the whole batch ran on the CPU, even with “Train on the AMD card” switched on. Now the batch’s RF-DETR models train on the card; only YOLO stays on the CPU.' }],
      es: [{ title: 'El entrenamiento por lotes con YOLO usa la tarjeta AMD para RF-DETR', text: 'Si un entrenamiento por lotes incluía un modelo YOLO, todo el lote se ejecutaba en la CPU, aunque «Entrenar en la tarjeta AMD» estuviera activado. Ahora los modelos RF-DETR del lote se entrenan en la tarjeta; solo YOLO sigue en la CPU.' }],
      fr: [{ title: 'L’entraînement par lots avec YOLO utilise la carte AMD pour RF-DETR', text: 'Lorsqu’un entraînement par lots comprenait un modèle YOLO, tout le lot tournait sur le CPU, même avec « Entraîner sur la carte AMD » activé. Désormais, les modèles RF-DETR du lot s’entraînent sur la carte ; seul YOLO reste sur le CPU.' }],
    },
  },
  {
    version: '0.31.2',
    items: {
      de: [{ title: 'Augmentation Studio erkennt Hallen in exportierten Datensätzen', text: 'Bei Datensätzen aus „Datensatz exportieren“ wurde jedes Bild zu einer eigenen Session, weil die Dateinamen absichtlich keine Videonamen enthalten. Jetzt gruppiert das Studio solche Bilder nach Bildgröße und Hallenhintergrund. Viele kleine Sessions erscheinen zusammengefasst, damit die Seite flüssig bleibt. Bitte im Studio einmal neu analysieren.' }],
      en: [{ title: 'Augmentation Studio finds venues in exported datasets', text: 'With datasets from “Export dataset”, every image became its own session, because the file names deliberately contain no video names. The studio now groups such images by image size and venue background. Many small sessions are shown as one list so the page stays fast. Please analyze once more in the studio.' }],
      es: [{ title: 'Augmentation Studio reconoce pabellones en conjuntos exportados', text: 'Con conjuntos de «Exportar conjunto de datos», cada imagen se convertía en su propia sesión, porque los nombres de archivo no incluyen a propósito nombres de vídeo. Ahora el estudio agrupa esas imágenes por tamaño y fondo del pabellón. Las sesiones pequeñas aparecen en una lista para que la página siga fluida. Vuelve a analizar una vez en el estudio.' }],
      fr: [{ title: 'Augmentation Studio reconnaît les salles des jeux exportés', text: 'Avec les jeux issus de « Exporter le jeu de données », chaque image devenait sa propre session, car les noms de fichiers ne contiennent volontairement aucun nom de vidéo. Le studio regroupe désormais ces images par taille et arrière-plan de la salle. Les nombreuses petites sessions apparaissent dans une liste pour que la page reste fluide. Veuillez relancer l’analyse une fois dans le studio.' }],
    },
  },
  {
    version: '0.31.1',
    items: {
      de: [{ title: 'Anleitung im Augmentation Studio', text: 'Beim Öffnen des Studios führt eine Schritt-für-Schritt-Anleitung mit Bildern durch alles: Quellordner, Einstellungen, Hallen und Teams prüfen, Erzeugen und der Vergleich mit „Nur echt“. „Nicht mehr anzeigen“ blendet sie aus, der Knopf „Anleitung“ im Studio holt sie zurück.' }],
      en: [{ title: 'Guide in the Augmentation Studio', text: 'When the studio opens, a step-by-step guide with pictures walks through everything: source folder, settings, checking venues and teams, generating and the comparison with “Real only”. “Don’t show again” hides it; the “Guide” button in the studio brings it back.' }],
      es: [{ title: 'Guía en Augmentation Studio', text: 'Al abrir el estudio, una guía paso a paso con imágenes lo explica todo: carpeta de origen, ajustes, revisión de pabellones y equipos, generación y la comparación con «Solo real». «No volver a mostrar» la oculta; el botón «Guía» del estudio la recupera.' }],
      fr: [{ title: 'Guide dans Augmentation Studio', text: 'À l’ouverture du studio, un guide pas à pas illustré explique tout : dossier source, réglages, vérification des salles et des équipes, génération et comparaison avec « Réel seul ». « Ne plus afficher » le masque ; le bouton « Guide » du studio le fait revenir.' }],
    },
  },
  {
    version: '0.31.0',
    items: {
      de: [
        { title: 'Training auf AMD-Grafikkarten (Windows)', text: 'Mit einer Radeon RX 7700 bis 7900 oder RX 9060/9070 erscheint „AMD-Training einrichten“. Danach trainiert „Lokal trainieren“ RF-DETR auf der Karte, bei einer RX 7800 XT etwa 40-mal schneller als auf der CPU. Braucht Python 3.12 und rund 6 GB.' },
        { title: 'Ein- und ausschaltbar', text: '„Auf der AMD-Karte trainieren“ schaltet zurück auf die CPU, falls etwas nicht klappt. Automatisches Markieren und Export laufen weiter wie bisher.' },
      ],
      en: [
        { title: 'Training on AMD graphics cards (Windows)', text: 'With a Radeon RX 7700 to 7900 or RX 9060/9070, “Set up AMD training” appears. Afterwards “Train locally” trains RF-DETR on the card, about 40 times faster than on the CPU on an RX 7800 XT. Needs Python 3.12 and about 6 GB.' },
        { title: 'Can be switched off', text: '“Train on the AMD card” switches back to the CPU if something goes wrong. Auto-labeling and export keep working as before.' },
      ],
      es: [
        { title: 'Entrenamiento en gráficas AMD (Windows)', text: 'Con una Radeon RX 7700 a 7900 o RX 9060/9070 aparece «Configurar entrenamiento AMD». Después, «Entrenar localmente» entrena RF-DETR en la tarjeta, unas 40 veces más rápido que en la CPU con una RX 7800 XT. Necesita Python 3.12 y unos 6 GB.' },
        { title: 'Se puede desactivar', text: '«Entrenar en la tarjeta AMD» vuelve a la CPU si algo falla. El marcado automático y la exportación siguen como antes.' },
      ],
      fr: [
        { title: 'Entraînement sur cartes graphiques AMD (Windows)', text: 'Avec une Radeon RX 7700 à 7900 ou RX 9060/9070, « Configurer l’entraînement AMD » apparaît. Ensuite, « Entraîner localement » entraîne RF-DETR sur la carte, environ 40 fois plus vite que sur le CPU avec une RX 7800 XT. Nécessite Python 3.12 et environ 6 Go.' },
        { title: 'Désactivable', text: '« Entraîner sur la carte AMD » revient au CPU en cas de problème. Le marquage automatique et l’export fonctionnent comme avant.' },
      ],
    },
  },
  {
    version: '0.30.6',
    items: {
      de: [{ title: 'AMD-Trainingstest repariert', text: '„Test AMD Training.bat“ brach nach dem Training mit einem Zeichenfehler der Windows-Konsole ab. Das Training selbst lief schon. Jetzt schreibt der Test seinen Bericht vollständig.' }],
      en: [{ title: 'AMD training test fixed', text: '“Test AMD Training.bat” stopped after training with a Windows console character error. Training itself already worked. The test now writes its full report.' }],
      es: [{ title: 'Prueba de entrenamiento AMD corregida', text: '«Test AMD Training.bat» se detenía tras el entrenamiento por un error de caracteres de la consola de Windows. El entrenamiento ya funcionaba. Ahora la prueba escribe su informe completo.' }],
      fr: [{ title: 'Test d’entraînement AMD corrigé', text: '« Test AMD Training.bat » s’arrêtait après l’entraînement à cause d’une erreur de caractères de la console Windows. L’entraînement fonctionnait déjà. Le test écrit maintenant son rapport complet.' }],
    },
  },
  {
    version: '0.30.5',
    items: {
      de: [
        { title: 'Leere Bilder nur nach Bestätigung im Training', text: 'Ein Bild ohne Boxen kommt erst ins Training, wenn du im Editor „Bild ist leer“ anklickst. Vorher zählten auch noch nicht markierte Bilder als leer und brachten dem Modell bei, dass Spieler und Ball Hintergrund sind. Die Übersicht zeigt, wie viele Bilder ohne Boxen draußen bleiben.' },
        { title: 'Leere Hallen als Hintergrundbilder', text: 'Fotos einer leeren Halle in den Arbeitsordner legen, vorbereiten, sichtbare Körbe markieren und „Bild ist leer“ bzw. die Boxen speichern. So lernt das Modell, was kein Ball und kein Spieler ist.' },
        { title: 'Klarere Rolle im Augmentation Studio', text: '„Statisch“ heißt jetzt „Bleibt an seinem Platz“: Körbe und Tore werden nicht verschoben, behalten aber ihre Klasse und Box.' },
      ],
      en: [
        { title: 'Empty images train only when confirmed', text: 'An image without boxes only goes into training once you click “Image is empty” in the editor. Before, images not yet labelled also counted as empty and taught the model that players and ball are background. The summary shows how many images without boxes stay out.' },
        { title: 'Empty halls as background images', text: 'Put photos of an empty hall into the working folder, prepare it, mark visible hoops and save, or click “Image is empty”. The model learns what is not a ball or a player.' },
        { title: 'Clearer role in the Augmentation Studio', text: '“Static” is now called “Stays in place”: hoops and goals are not moved but keep their class and box.' },
      ],
      es: [
        { title: 'Imágenes vacías solo si se confirman', text: 'Una imagen sin cuadros solo entra en el entrenamiento si pulsas «La imagen está vacía» en el editor. Antes también contaban como vacías las imágenes aún sin etiquetar y enseñaban al modelo que jugadores y balón son fondo. El resumen muestra cuántas imágenes sin cuadros quedan fuera.' },
        { title: 'Pabellones vacíos como fondo', text: 'Pon fotos de un pabellón vacío en la carpeta de trabajo, prepárala, marca las canastas visibles y guarda, o pulsa «La imagen está vacía». El modelo aprende qué no es un balón ni un jugador.' },
        { title: 'Función más clara en Augmentation Studio', text: '«Estático» ahora se llama «Se queda en su sitio»: canastas y porterías no se mueven, pero conservan su clase y su cuadro.' },
      ],
      fr: [
        { title: 'Images vides seulement après confirmation', text: 'Une image sans boîtes n’entre dans l’entraînement qu’après un clic sur « L’image est vide » dans l’éditeur. Avant, les images pas encore annotées comptaient aussi comme vides et apprenaient au modèle que joueurs et ballon sont du fond. Le résumé indique combien d’images sans boîtes restent dehors.' },
        { title: 'Salles vides comme images de fond', text: 'Placez des photos d’une salle vide dans le dossier de travail, préparez-le, annotez les paniers visibles et enregistrez, ou cliquez sur « L’image est vide ». Le modèle apprend ce qui n’est ni un ballon ni un joueur.' },
        { title: 'Rôle plus clair dans Augmentation Studio', text: '« Statique » s’appelle désormais « Reste à sa place » : paniers et buts ne sont pas déplacés mais gardent leur classe et leur boîte.' },
      ],
    },
  },
  {
    version: '0.30.4',
    items: {
      de: [
        { title: 'Augmentation Studio erklärt, was fehlt', text: '„Quellordner wählen“ ist gesperrt, solange kein Arbeitsordner vorbereitet oder ML nicht eingerichtet ist. Daneben steht, was zuerst zu tun ist. Vorher passierte beim Klick scheinbar nichts.' },
      ],
      en: [
        { title: 'Augmentation Studio says what is missing', text: '“Select source folder” stays disabled until a working folder is prepared and ML is set up, and the text next to it says what to do first. Before, clicking it seemed to do nothing.' },
      ],
      es: [
        { title: 'Augmentation Studio indica lo que falta', text: '«Seleccionar carpeta de origen» queda desactivado hasta que haya una carpeta de trabajo preparada y el ML esté configurado; al lado se indica qué hacer primero. Antes, al pulsarlo no parecía pasar nada.' },
      ],
      fr: [
        { title: 'Augmentation Studio indique ce qui manque', text: '« Sélectionner le dossier source » reste désactivé tant qu’aucun dossier de travail n’est préparé et que le ML n’est pas configuré ; le texte à côté indique quoi faire d’abord. Avant, le clic semblait sans effet.' },
      ],
    },
  },
  {
    version: '0.30.3',
    items: {
      de: [
        { title: 'Vorschläge auch für Testbilder', text: '„Vorschläge berechnen“ in der Konsistenzprüfung nimmt jetzt dieselben markierten Bilder wie die Prüfliste, auch die des unabhängigen Testsets. Vorher kam „Keine markierten Bilder“.' },
        { title: 'Klare Meldungen statt Diagnosebericht', text: 'Hinweise wie „Keine markierten Bilder“ erscheinen jetzt als solche statt als „mit Code 1 beendet“ und starten keinen Grafikkarten-Diagnosebericht mehr.' },
        { title: 'AMD-Training testen (Windows)', text: 'Im Ordner tools liegt „Test AMD Training.bat“. Es prüft in einem eigenen Ordner, ob deine AMD-Karte mit AMDs ROCm-PyTorch RF-DETR trainieren kann, und schreibt einen Bericht auf den Desktop.' },
      ],
      en: [
        { title: 'Suggestions for test images too', text: '“Compute suggestions” in the consistency review now takes the same flagged images as the review list, including those of the independent test set. Before, it said “No flagged images”.' },
        { title: 'Clear messages instead of a diagnosis report', text: 'Notes such as “No flagged images” now appear as they are instead of “exited with code 1”, and no longer start a graphics card diagnosis.' },
        { title: 'Test AMD training (Windows)', text: 'The tools folder contains “Test AMD Training.bat”. In a separate folder it checks whether your AMD card can train RF-DETR with AMD’s ROCm build of PyTorch and writes a report to the desktop.' },
      ],
      es: [
        { title: 'Sugerencias también para imágenes de prueba', text: '«Calcular sugerencias» en la revisión de coherencia usa ahora las mismas imágenes marcadas que la lista de revisión, también las del conjunto de prueba independiente. Antes decía «No hay imágenes marcadas».' },
        { title: 'Mensajes claros en lugar de informe', text: 'Avisos como «No hay imágenes marcadas» aparecen ahora tal cual en lugar de «terminó con el código 1» y ya no inician un diagnóstico de la gráfica.' },
        { title: 'Probar el entrenamiento con AMD (Windows)', text: 'La carpeta tools contiene «Test AMD Training.bat». En una carpeta aparte comprueba si tu tarjeta AMD puede entrenar RF-DETR con PyTorch ROCm de AMD y deja un informe en el escritorio.' },
      ],
      fr: [
        { title: 'Suggestions aussi pour les images de test', text: '« Calculer les suggestions » dans la revue de cohérence prend maintenant les mêmes images signalées que la liste de revue, y compris celles du jeu de test indépendant. Avant, le message était « Aucune image signalée ».' },
        { title: 'Messages clairs au lieu d’un rapport', text: 'Des messages comme « Aucune image signalée » apparaissent tels quels au lieu de « terminé avec le code 1 » et ne lancent plus de diagnostic de la carte graphique.' },
        { title: 'Tester l’entraînement AMD (Windows)', text: 'Le dossier tools contient « Test AMD Training.bat ». Dans un dossier séparé, il vérifie si votre carte AMD peut entraîner RF-DETR avec PyTorch ROCm d’AMD et écrit un rapport sur le bureau.' },
      ],
    },
  },
  {
    version: '0.30.2',
    items: {
      de: [
        { title: 'Spielfeldgrenze entfernen', text: 'Im Dialog „Spielfeld bearbeiten“ löscht „Spielfeldgrenze entfernen“ die Grenze dieses Videos, mit Häkchen „Für alle Videos übernehmen“ die aller Videos. Zur Sicherheit braucht es einen zweiten Klick. Automatisch markieren filtert danach nicht mehr nach Spielfeld.' },
        { title: 'Arbeitsspeicher unter Windows', text: 'Die Hardware-Zeile zeigt unter Windows jetzt den Arbeitsspeicher statt „? GB“. Die Erkennung auf AMD- und Intel-Grafik heißt nicht mehr „experimentell“.' },
      ],
      en: [
        { title: 'Remove a field boundary', text: 'In “Edit field boundaries”, “Remove field boundary” deletes this video’s boundary, or every video’s with “Use for all videos” ticked. It asks for a second click to be safe. Auto-label then no longer filters by field.' },
        { title: 'Memory on Windows', text: 'The hardware line on Windows now shows the memory instead of “? GB”. Detection on AMD and Intel graphics is no longer called “experimental”.' },
      ],
      es: [
        { title: 'Eliminar el límite del campo', text: 'En «Editar el campo», «Eliminar límite del campo» borra el límite de este vídeo, o el de todos con «Usar para todos los vídeos» marcado. Por seguridad pide un segundo clic. Después, el marcado automático ya no filtra por campo.' },
        { title: 'Memoria en Windows', text: 'La línea de hardware en Windows muestra ahora la memoria en lugar de «? GB». La detección en gráficas AMD e Intel ya no se llama «experimental».' },
      ],
      fr: [
        { title: 'Supprimer la limite du terrain', text: 'Dans « Modifier le terrain », « Supprimer la limite du terrain » efface la limite de cette vidéo, ou de toutes avec « Utiliser pour toutes les vidéos » coché. Un second clic confirme. Le marquage automatique ne filtre alors plus par terrain.' },
        { title: 'Mémoire sous Windows', text: 'La ligne matériel sous Windows affiche maintenant la mémoire au lieu de « ? GB ». La détection sur cartes AMD et Intel n’est plus dite « expérimentale ».' },
      ],
    },
  },
  {
    version: '0.30.1',
    items: {
      de: [
        { title: 'AMD-Grafik bestätigt', text: 'Ein Tester hat die Erkennung auf einer AMD Radeon RX 7800 XT unter Windows 11 geprüft: 6,7 ms statt 110,7 ms pro Bild auf der CPU, also etwa 16-mal schneller, mit denselben Boxen. Training läuft auf AMD weiterhin auf der CPU.' },
        { title: 'Genauerer Diagnosebericht', text: 'Der Bericht warnt nicht mehr fälschlich, wenn dxdiag „Driver Problem Code: Unknown“ meldet, und nennt unter Windows und Linux die Reco-Trainer-Version. Nach einem YOLO-Training bleibt kein leerer Ordner „runs“ mehr zurück.' },
      ],
      en: [
        { title: 'AMD graphics confirmed', text: 'A tester checked detection on an AMD Radeon RX 7800 XT with Windows 11: 6.7 ms instead of 110.7 ms per image on the CPU, about 16 times faster, with the same boxes. Training on AMD still runs on the CPU.' },
        { title: 'More accurate diagnosis report', text: 'The report no longer warns by mistake when dxdiag says “Driver Problem Code: Unknown”, and it names the Reco Trainer version on Windows and Linux. YOLO training no longer leaves an empty “runs” folder behind.' },
      ],
      es: [
        { title: 'Gráficos AMD confirmados', text: 'Un probador comprobó la detección en una AMD Radeon RX 7800 XT con Windows 11: 6,7 ms en lugar de 110,7 ms por imagen en la CPU, unas 16 veces más rápido, con las mismas cajas. El entrenamiento en AMD sigue en la CPU.' },
        { title: 'Informe de diagnóstico más preciso', text: 'El informe ya no avisa por error cuando dxdiag indica «Driver Problem Code: Unknown» y muestra la versión de Reco Trainer en Windows y Linux. El entrenamiento YOLO ya no deja una carpeta «runs» vacía.' },
      ],
      fr: [
        { title: 'Carte AMD confirmée', text: 'Un testeur a vérifié la détection sur une AMD Radeon RX 7800 XT sous Windows 11 : 6,7 ms au lieu de 110,7 ms par image sur le CPU, environ 16 fois plus rapide, avec les mêmes boîtes. L’entraînement sur AMD reste sur le CPU.' },
        { title: 'Rapport de diagnostic plus précis', text: 'Le rapport n’avertit plus à tort quand dxdiag indique « Driver Problem Code: Unknown » et affiche la version de Reco Trainer sous Windows et Linux. L’entraînement YOLO ne laisse plus de dossier « runs » vide.' },
      ],
    },
  },
  {
    version: '0.30.0',
    items: {
      de: [
        { title: 'Neuer Look, hell und dunkel', text: 'Die Oberfläche sieht jetzt aus wie eine Mac-App: Systemschrift, Milchglas-Seitenleiste, ruhige Karten und Dialoge. Das Bild steht auf einer dunklen Bühne mit Glasleiste. Der Dunkelmodus folgt der Systemeinstellung. Alle Knöpfe sind dort, wo sie waren.' },
        { title: 'YOLO trainieren (optional)', text: 'Im Dialog „Lokal trainieren“ gibt es neben RF-DETR jetzt YOLO26 Nano, Small und Medium, auch beide im Vergleich. YOLO wird beim ersten Mal installiert, nachdem du der AGPL-3.0-Lizenz von Ultralytics zugestimmt hast; damit trainierte Modelle stehen ebenfalls unter AGPL-3.0. Das Modell kann danach automatisch markieren, getestet und in der Tracking-Simulation genutzt werden.' },
        { title: 'Bildgröße für kleine Bälle', text: 'Für YOLO lässt sich die Bildgröße wählen (640, 960 oder 1280 px). Größer findet kleine Bälle besser, braucht aber mehr Zeit und Speicher.' },
      ],
      en: [
        { title: 'New look, light and dark', text: 'The interface now looks like a Mac app: system font, translucent sidebar, calm cards and dialogs. The frame sits on a dark stage with a glass bar. Dark mode follows the system setting. Every button is where it was.' },
        { title: 'Train YOLO (optional)', text: 'The “Train locally” dialog now offers YOLO26 Nano, Small and Medium next to RF-DETR, also both side by side. YOLO is installed the first time, after you accept the Ultralytics AGPL-3.0 license; models trained with it are AGPL-3.0 as well. The model can then auto-label, be tested and be used in the tracking simulation.' },
        { title: 'Image size for small balls', text: 'For YOLO you can choose the image size (640, 960 or 1280 px). Larger finds small balls better but needs more time and memory.' },
      ],
      es: [
        { title: 'Nuevo aspecto, claro y oscuro', text: 'La interfaz parece ahora una app de Mac: fuente del sistema, barra lateral translúcida, tarjetas y diálogos tranquilos. La imagen está sobre un escenario oscuro con una barra de cristal. El modo oscuro sigue el ajuste del sistema. Todos los botones siguen en su sitio.' },
        { title: 'Entrenar YOLO (opcional)', text: 'El diálogo «Entrenar localmente» ofrece ahora YOLO26 Nano, Small y Medium junto a RF-DETR, también ambos para comparar. YOLO se instala la primera vez, tras aceptar la licencia AGPL-3.0 de Ultralytics; los modelos entrenados con él también son AGPL-3.0. Después el modelo puede marcar automáticamente, probarse y usarse en la simulación de seguimiento.' },
        { title: 'Tamaño de imagen para balones pequeños', text: 'Para YOLO se puede elegir el tamaño de imagen (640, 960 o 1280 px). Más grande detecta mejor balones pequeños, pero necesita más tiempo y memoria.' },
      ],
      fr: [
        { title: 'Nouveau look, clair et sombre', text: 'L’interface ressemble maintenant à une app Mac : police système, barre latérale translucide, cartes et dialogues sobres. L’image repose sur une scène sombre avec une barre en verre. Le mode sombre suit le réglage du système. Tous les boutons sont restés à leur place.' },
        { title: 'Entraîner YOLO (optionnel)', text: 'La boîte « Entraîner localement » propose maintenant YOLO26 Nano, Small et Medium à côté de RF-DETR, aussi les deux en comparaison. YOLO est installé la première fois, après acceptation de la licence AGPL-3.0 d’Ultralytics ; les modèles entraînés avec lui sont aussi sous AGPL-3.0. Le modèle peut ensuite marquer automatiquement, être testé et servir dans la simulation de suivi.' },
        { title: 'Taille d’image pour les petits ballons', text: 'Pour YOLO, on peut choisir la taille d’image (640, 960 ou 1280 px). Plus grand détecte mieux les petits ballons, mais demande plus de temps et de mémoire.' },
      ],
    },
  },
  {
    version: '0.29.0',
    items: {
      de: [
        { title: 'Größeres Bild auf Laptops', text: 'Das Bildfenster nutzt jetzt fast die ganze Höhe (bei 1366×768 mehr als doppelt so hoch). Der Trainingsbereich liegt darunter und lässt sich wie bisher einklappen.' },
        { title: 'Tastenkürzel', text: 'Leertaste gedrückt halten = Bild verschieben, danach ist „Neue Box“ wieder aktiv. B = Neue Box, 1–9 = Klasse in der Reihenfolge der Liste, ←/→ = Bild wechseln. In Eingabefeldern lösen die Kürzel nicht aus.' },
        { title: 'Zoom zum Mauszeiger, bis 20×', text: 'Das Mausrad zoomt zu dem Punkt unter dem Mauszeiger, auch während „Neue Box“ aktiv ist. Maximal 20× statt 5×, in gleichmäßigen Schritten. Ab etwa 3× sind die echten Bildpunkte des Balls zu sehen.' },
        { title: 'Feine Linien bei jedem Zoom', text: 'Rahmen, Anfasser und Beschriftungen bleiben auf dem Bildschirm gleich dünn. Die gespeicherten Box-Koordinaten ändern sich dadurch nicht.' },
        { title: 'Spielfeld pro Video', text: 'Jedes Video (jede Kameraposition) hat jetzt eine eigene Spielfeldgrenze; „Für alle Videos übernehmen“ setzt sie auf einmal. Bestehende Projekte übernehmen die bisherige Grenze automatisch für alle Videos, vorher wird eine Sicherung angelegt.' },
        { title: '„Modelle testen“ erklärt sich', text: 'Ist der Vergleich noch nicht möglich, steht dabei, was fehlt (z. B. „Noch kein trainiertes Modell“, „Noch kein bestätigtes Testset“) und was als Nächstes zu tun ist. Deaktivierte Knöpfe nennen den Grund.' },
        { title: 'AMD- und Intel-Grafik (Windows, experimentell)', text: 'Ohne NVIDIA-GPU fragt Reco Trainer, ob die Grafikkarte versucht werden soll. Klappt es nicht, öffnet sich ein Diagnosebericht zum Kopieren oder Herunterladen. Fehlermeldungen wie „mit Code 1 beendet“ nennen jetzt die Ursache.' },
      ],
      en: [
        { title: 'Larger frame on laptops', text: 'The frame viewer now uses almost the full height (more than twice as high at 1366×768). The training section sits below it and can still be collapsed.' },
        { title: 'Keyboard shortcuts', text: 'Hold space = pan, afterwards “New box” is active again. B = New box, 1–9 = class in list order, ←/→ = change frame. Shortcuts do not fire while you type in a field.' },
        { title: 'Zoom to the mouse, up to 20×', text: 'The mouse wheel zooms to the point under the mouse, also while “New box” is active. Up to 20× instead of 5×, in even steps. From about 3× you see the ball’s real pixels.' },
        { title: 'Thin lines at every zoom', text: 'Box outlines, handles and labels keep their size on screen. Stored box coordinates do not change.' },
        { title: 'Field boundary per video', text: 'Every video (every camera position) now has its own field boundary; “Use for all videos” sets it at once. Existing projects copy their previous boundary to all videos automatically, after a backup.' },
        { title: '“Test models” explains itself', text: 'When the comparison cannot run yet, it says what is missing (e.g. “No trained model”, “No confirmed test set”) and what to do next. Disabled buttons name the reason.' },
        { title: 'AMD and Intel graphics (Windows, experimental)', text: 'Without an NVIDIA GPU, Reco Trainer asks whether to try the graphics card. If it fails, a diagnosis report opens to copy or download. Errors like “exited with code 1” now name the cause.' },
      ],
      es: [
        { title: 'Imagen más grande en portátiles', text: 'El visor usa ahora casi toda la altura (más del doble a 1366×768). La zona de entrenamiento queda debajo y se puede seguir plegando.' },
        { title: 'Atajos de teclado', text: 'Mantener la barra espaciadora = mover la imagen; después «Nuevo cuadro» vuelve a estar activo. B = Nuevo cuadro, 1–9 = clase en el orden de la lista, ←/→ = cambiar imagen. No se activan mientras escribes en un campo.' },
        { title: 'Zoom hacia el cursor, hasta 20×', text: 'La rueda del ratón amplía hacia el punto bajo el cursor, también con «Nuevo cuadro» activo. Hasta 20× en lugar de 5×, en pasos uniformes. A partir de unos 3× se ven los píxeles reales del balón.' },
        { title: 'Líneas finas con cualquier zoom', text: 'Los bordes, tiradores y etiquetas mantienen su tamaño en pantalla. Las coordenadas guardadas no cambian.' },
        { title: 'Límite de campo por vídeo', text: 'Cada vídeo (cada posición de cámara) tiene ahora su propio límite de campo; «Usar para todos los vídeos» lo aplica de una vez. Los proyectos existentes copian su límite anterior a todos los vídeos automáticamente, tras una copia de seguridad.' },
        { title: '«Probar modelos» se explica', text: 'Si la comparación aún no es posible, indica qué falta (p. ej. «Aún no hay un modelo entrenado», «Aún no hay un conjunto de prueba confirmado») y qué hacer después. Los botones desactivados indican el motivo.' },
        { title: 'Gráficas AMD e Intel (Windows, experimental)', text: 'Sin GPU NVIDIA, Reco Trainer pregunta si probar la gráfica. Si falla, se abre un informe de diagnóstico para copiar o descargar. Errores como «terminó con el código 1» indican ahora la causa.' },
      ],
      fr: [
        { title: 'Image plus grande sur les portables', text: 'La visionneuse utilise maintenant presque toute la hauteur (plus du double en 1366×768). La zone d’entraînement se trouve dessous et peut toujours être repliée.' },
        { title: 'Raccourcis clavier', text: 'Maintenir la barre d’espace = déplacer l’image ; ensuite « Nouvelle boîte » est de nouveau active. B = Nouvelle boîte, 1–9 = classe dans l’ordre de la liste, ←/→ = changer d’image. Inactifs pendant la saisie dans un champ.' },
        { title: 'Zoom vers la souris, jusqu’à 20×', text: 'La molette zoome vers le point sous la souris, même avec « Nouvelle boîte » active. Jusqu’à 20× au lieu de 5×, par paliers réguliers. À partir d’environ 3×, on voit les vrais pixels du ballon.' },
        { title: 'Traits fins à tout zoom', text: 'Contours, poignées et étiquettes gardent leur taille à l’écran. Les coordonnées enregistrées ne changent pas.' },
        { title: 'Limite de terrain par vidéo', text: 'Chaque vidéo (chaque position de caméra) a maintenant sa propre limite de terrain ; « Utiliser pour toutes les vidéos » l’applique d’un coup. Les projets existants copient automatiquement leur limite précédente sur toutes les vidéos, après une sauvegarde.' },
        { title: '« Tester les modèles » s’explique', text: 'Si la comparaison n’est pas encore possible, il indique ce qui manque (p. ex. « Pas encore de modèle entraîné », « Pas encore de jeu de test confirmé ») et quoi faire ensuite. Les boutons désactivés indiquent la raison.' },
        { title: 'Cartes AMD et Intel (Windows, expérimental)', text: 'Sans GPU NVIDIA, Reco Trainer demande s’il faut essayer la carte graphique. En cas d’échec, un rapport de diagnostic s’ouvre pour le copier ou le télécharger. Les erreurs comme « terminé avec le code 1 » nomment maintenant la cause.' },
      ],
    },
  },
];

const heading: Record<Language, (version: string) => string> = {
  de: (version) => `Neu in Reco Trainer ${version}`,
  en: (version) => `What’s new in Reco Trainer ${version}`,
  es: (version) => `Novedades de Reco Trainer ${version}`,
  fr: (version) => `Nouveautés de Reco Trainer ${version}`,
};
const okLabel: Record<Language, string> = { de: 'Los geht’s', en: 'Got it', es: 'Entendido', fr: 'C’est parti' };

export function whatsNewFor(version: string) {
  return WHATS_NEW.find((entry) => entry.version === version) ?? null;
}

export default function WhatsNew({ language, version, onClose }: { language: Language; version: string; onClose: () => void }) {
  const entry = whatsNewFor(version);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  if (!entry) return null;
  return <div className="walkthrough-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="whats-new-dialog" role="dialog" aria-modal="true" aria-labelledby="whats-new-title" tabIndex={-1} ref={(node) => node?.focus({ preventScroll: true })}>
      <h2 id="whats-new-title">✦ {heading[language](entry.version)}</h2>
      <ul>{entry.items[language].map((item) => <li key={item.title}><strong>{item.title}</strong><span>{item.text}</span></li>)}</ul>
      <div className="whats-new-actions"><button data-hint={hintText[language].close} type="button" className="primary-button" onClick={onClose}>{okLabel[language]}</button></div>
    </section>
  </div>;
}
