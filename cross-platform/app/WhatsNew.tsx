import { useEffect } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type Item = { title: string; text: string };

/** Shown once after an update (see page.tsx, key reco-whats-new-seen-v1); clicking the version
 *  number in the header opens it again. Add the next version's entry at the top. */
export const WHATS_NEW: { version: string; items: Record<Language, Item[]> }[] = [
  {
    version: '0.37.0',
    items: {
      de: [
        { title: 'Rolle je Video wählen', text: 'Unter den Chips der Bilderliste legst du für jedes Video Training, Validierung oder Test fest. Ein Video bleibt ganz in seiner Rolle. Fehlt Validierung oder Test, sucht die App sie aus den übrigen Videos aus; ein Video bleibt zum Trainieren.' },
        { title: 'YOLO26-Modelle im Model Hub', text: 'Mit YOLO trainierte Modelle (Nano, Small, Medium) lassen sich wie RF-DETR im Hub teilen und laden, mit demselben Sicherheitskonzept: nur bereinigte Gewichte als safetensors, keine Bilder, Prüfung durch einen Maintainer. Beim Laden wird das Modell lokal aus den Gewichten neu aufgebaut und als ONNX exportiert. YOLO26 steht unter der AGPL-3.0; die Lizenz wird angezeigt und bestätigt.' },
      ],
      en: [
        { title: 'Choose the role of each video', text: 'Under the chips of the image list you set training, validation or test for each video. A video stays whole in its role. If validation or test is missing, the app picks it from the remaining videos; one video is left for training.' },
        { title: 'YOLO26 models in the Model Hub', text: 'Models trained with YOLO (Nano, Small, Medium) can be shared and loaded in the hub like RF-DETR, with the same safety concept: only cleaned weights as safetensors, no pictures, review by a maintainer. When loaded, the model is rebuilt locally from the weights and exported as ONNX. YOLO26 is under the AGPL-3.0; the license is shown and confirmed.' },
      ],
      es: [
        { title: 'Elige el rol de cada vídeo', text: 'Bajo los chips de la lista de imágenes fijas entrenamiento, validación o prueba para cada vídeo. Un vídeo queda entero en su rol. Si falta validación o prueba, la app la elige entre los demás vídeos; uno queda para entrenar.' },
        { title: 'Modelos YOLO26 en el Model Hub', text: 'Los modelos entrenados con YOLO (Nano, Small, Medium) se pueden compartir y cargar en el hub como RF-DETR, con el mismo concepto de seguridad: solo pesos limpios como safetensors, sin imágenes, revisión por un mantenedor. Al cargarlo, el modelo se reconstruye localmente a partir de los pesos y se exporta como ONNX. YOLO26 está bajo la AGPL-3.0; la licencia se muestra y se confirma.' },
      ],
      fr: [
        { title: 'Choisir le rôle de chaque vidéo', text: 'Sous les puces de la liste d’images, vous fixez entraînement, validation ou test pour chaque vidéo. Une vidéo reste entière dans son rôle. S’il manque la validation ou le test, l’app les choisit parmi les autres vidéos ; une vidéo reste pour l’entraînement.' },
        { title: 'Modèles YOLO26 dans le Model Hub', text: 'Les modèles entraînés avec YOLO (Nano, Small, Medium) se partagent et se chargent dans le hub comme RF-DETR, avec le même concept de sécurité : seulement des poids nettoyés en safetensors, pas d’images, relecture par un mainteneur. Au chargement, le modèle est reconstruit localement à partir des poids et exporté en ONNX. YOLO26 est sous AGPL-3.0 ; la licence est affichée et confirmée.' },
      ],
    },
  },
  {
    version: '0.36.0',
    items: {
      de: [
        { title: 'Simulation: Länge frei wählen', text: 'Die Länge des Abschnitts ist jetzt eine freie Zahl in Sekunden. Mit ¼, ½, ¾ und „ganz“ nimmst du diesen Anteil des Videos. Die Grenze von 2160 Bildern ist weg; erst über 20.000 Bilder gibt es einen Hinweis. Ändern sich Start, Länge oder Dichte, fragt die Simulation, ob sie den neuen Ausschnitt jetzt analysieren soll, ohne das Video neu zu laden.' },
        { title: 'Wie viele neue Bilder sind Test?', text: 'Vor dem Training fragt der Dialog, wie viele der seit dem letzten Training bestätigten Bilder als Test dienen und nicht trainiert werden (keine, 10, 20, 30 oder 50 %). Die Auswahl bleibt bei späteren Trainings Test.' },
        { title: 'Bilder ohne Markierung nachlabeln', text: 'Im Trainingsdialog führt „Bilder ohne Markierung nachlabeln“ direkt zu den Bildern, die noch nicht im Training sind.' },
      ],
      en: [
        { title: 'Simulation: choose any length', text: 'The length of the section is now a free number of seconds. ¼, ½, ¾ and “all” take that share of the video. The limit of 2,160 images is gone; only above 20,000 images you get a notice. If you change the start, length or density, the simulation asks whether to analyse the new section now, without loading the video again.' },
        { title: 'How many new images are test?', text: 'Before training, the dialog asks how many of the images confirmed since the last training should serve as test and not be trained on (none, 10, 20, 30 or 50 %). The choice stays test in later trainings.' },
        { title: 'Label the images without boxes', text: 'In the training dialog, “Label the images without boxes” takes you straight to the images that are not in training yet.' },
      ],
      es: [
        { title: 'Simulación: elige cualquier duración', text: 'La duración del tramo es ahora un número libre de segundos. ¼, ½, ¾ y «todo» toman esa parte del vídeo. El límite de 2160 imágenes ya no existe; solo por encima de 20.000 imágenes aparece un aviso. Si cambias el inicio, la duración o la densidad, la simulación pregunta si analizar ahora el nuevo tramo, sin volver a cargar el vídeo.' },
        { title: '¿Cuántas imágenes nuevas son de prueba?', text: 'Antes de entrenar, el cuadro pregunta cuántas de las imágenes confirmadas desde el último entrenamiento deben servir de prueba y no entrenarse (ninguna, 10, 20, 30 o 50 %). La elección sigue siendo de prueba en entrenamientos posteriores.' },
        { title: 'Marcar las imágenes sin cuadros', text: 'En el cuadro de entrenamiento, «Marcar las imágenes sin cuadros» lleva directamente a las imágenes que aún no están en el entrenamiento.' },
      ],
      fr: [
        { title: 'Simulation : choisir une durée libre', text: 'La durée du passage est maintenant un nombre libre de secondes. ¼, ½, ¾ et « tout » prennent cette part de la vidéo. La limite de 2160 images disparaît ; un avis n’apparaît qu’au-delà de 20 000 images. Si vous changez le début, la durée ou la densité, la simulation demande si elle doit analyser le nouveau passage maintenant, sans recharger la vidéo.' },
        { title: 'Combien de nouvelles images servent de test ?', text: 'Avant l’entraînement, la fenêtre demande combien des images confirmées depuis le dernier entraînement doivent servir de test et ne pas être entraînées (aucune, 10, 20, 30 ou 50 %). Le choix reste un test aux entraînements suivants.' },
        { title: 'Annoter les images sans boîtes', text: 'Dans la fenêtre d’entraînement, « Annoter les images sans boîtes » mène directement aux images qui ne sont pas encore dans l’entraînement.' },
      ],
    },
  },
  {
    version: '0.35.0',
    items: {
      de: [
        { title: 'Simulation: Abschnitt und Bilddichte wählen', text: 'Die Balltracking-Simulation hat keine feste Obergrenze von 300 Bildern mehr. Du wählst Start, Länge und ob alle 12 Bilder pro Sekunde oder nur jedes n-te ausgewertet werden (bis 2160 Bilder je Lauf). Die Vorausschau zählt in Sekunden, ein Hinweis warnt bei sehr dünner Dichte, vor dem Start wird der freie Speicher geprüft, und Start, Länge und Dichte merkt sich die Simulation je Projekt. Die Oberfläche zeigt, wie viele Bilder das ergibt; „Gleiches Video wie A“ nimmt denselben Abschnitt.' },
        { title: 'Weitere Bälle werden vorgemerkt', text: 'Beim Übernehmen von Szenen als Prüfkandidaten wird jedes Bild noch einmal auf weitere Bälle geprüft, etwa am Rand. Sie kommen als Vorschläge ins Bild; „Ball“ bestätigt alle Ball-Boxen, also prüfe sie kurz.' },
        { title: 'Nach Ball / Kein Ball sofort weiter', text: 'Nach „Ball“, „Kein Ball“ und „Überspringen“ öffnet sich direkt das nächste Prüfbild. Vorher konnte die Auswahl nach dem Speichern an den Anfang der Liste springen.' },
      ],
      en: [
        { title: 'Simulation: choose the section and the density', text: 'The ball-tracking simulation no longer has a fixed limit of 300 images. You choose the start, the length, and whether to analyse all 12 images per second or only every n-th (up to 2,160 images per run). The look-ahead counts in seconds, a note warns about very sparse images, free disk space is checked before the start, and start, length and density are remembered per project. The panel shows how many images that gives; “Same video as A” uses the same section.' },
        { title: 'Further balls are suggested', text: 'When you add scenes as review candidates, every image is checked once more for further balls, for example at the edge. They come in as suggested boxes; “Ball” confirms all ball boxes, so check them briefly.' },
        { title: 'Straight on after Ball / No ball', text: 'After “Ball”, “No ball” and “Skip” the next review image opens. Before, the selection could jump back to the start of the list after saving.' },
      ],
      es: [
        { title: 'Simulación: elige el tramo y la densidad', text: 'La simulación de seguimiento del balón ya no tiene un límite fijo de 300 imágenes. Eliges el inicio, la duración y si se analizan las 12 imágenes por segundo o solo una de cada n (hasta 2160 imágenes por ejecución). La anticipación cuenta en segundos, un aviso advierte de una densidad muy baja antes del inicio se comprueba el espacio libre y la simulación recuerda inicio, duración y densidad por proyecto. El panel muestra cuántas imágenes salen; «Mismo vídeo que A» usa el mismo tramo.' },
        { title: 'Se proponen más balones', text: 'Al añadir escenas como candidatas, cada imagen se comprueba otra vez en busca de más balones, por ejemplo en el borde. Aparecen como cuadros propuestos; «Balón» confirma todos los cuadros de balón, así que revísalos brevemente.' },
        { title: 'Directo a la siguiente tras Balón / No es balón', text: 'Tras «Balón», «No es balón» y «Omitir» se abre la siguiente imagen de revisión. Antes la selección podía saltar al principio de la lista tras guardar.' },
      ],
      fr: [
        { title: 'Simulation : choisir le passage et la densité', text: 'La simulation de suivi du ballon n’a plus de limite fixe de 300 images. Vous choisissez le début, la durée et si les 12 images par seconde ou seulement une sur n sont analysées (jusqu’à 2160 images par exécution). L’anticipation compte en secondes, un avis prévient d’une densité très faible l’espace libre est vérifié avant le départ, et la simulation retient début, durée et densité par projet. Le panneau indique le nombre d’images ; « Même vidéo que A » reprend le même passage.' },
        { title: 'D’autres ballons sont proposés', text: 'Quand vous ajoutez des scènes comme candidates, chaque image est revérifiée pour d’autres ballons, par exemple en bordure. Ils arrivent comme boîtes proposées ; « Ballon » confirme toutes les boîtes de ballon, vérifiez-les donc rapidement.' },
        { title: 'Directement la suivante après Ballon / Pas de ballon', text: 'Après « Ballon », « Pas de ballon » et « Ignorer », l’image suivante à vérifier s’ouvre. Avant, la sélection pouvait revenir au début de la liste après l’enregistrement.' },
      ],
    },
  },
  {
    version: '0.34.0',
    items: {
      de: [
        { title: 'Simulation: Fortschritt in Schritten, Abbruch mit Wahl', text: 'Die Balltracking-Simulation zeigt jetzt drei Schritte mit Zähler und Restzeit: Bilder extrahieren, Modell laden, Ball suchen. Beim Abbrechen wählst du, ob die schon ausgewerteten Bilder als kürzere Simulation bleiben oder alle Bilder des Laufs gelöscht werden.' },
        { title: 'Auffällige Szenen anspringen und auswählen', text: 'Nach dem Lauf listet die Simulation Szenen, in denen das Modell wahrscheinlich danebenliegt: Lücke (Ball fehlt kurz), Sprung (Fund springt weiter, als ein Ball fliegt), Unsicher und Verloren. Ein Klick springt zur Szene. Angekreuzte Szenen kommen mit wenigen Bildern als Prüfkandidaten in die Bilderliste, wo du jedes Bild einzeln prüfst. Lücken und Sprünge sind vorab angekreuzt.' },
      ],
      en: [
        { title: 'Simulation: progress in steps, cancel with a choice', text: 'The ball-tracking simulation now shows three steps with a counter and the time left: extract images, load the model, find the ball. When you cancel, you choose whether the images analysed so far stay as a shorter simulation or all images of the run are deleted.' },
        { title: 'Jump to notable scenes and choose them', text: 'After the run the simulation lists scenes where the model is probably wrong: gap (ball briefly missing), jump (the detection moves further than a ball travels), unsure and lost. A click jumps to the scene. Ticked scenes go into the image list as review candidates, a few images each, where you check every image yourself. Gaps and jumps are ticked to begin with.' },
      ],
      es: [
        { title: 'Simulación: progreso por pasos y cancelación con elección', text: 'La simulación de seguimiento del balón muestra ahora tres pasos con contador y tiempo restante: extraer imágenes, cargar el modelo, buscar el balón. Al cancelar eliges si las imágenes ya analizadas se conservan como una simulación más corta o si se eliminan todas las imágenes de la ejecución.' },
        { title: 'Saltar a las escenas llamativas y elegirlas', text: 'Tras la ejecución, la simulación enumera las escenas en las que el modelo probablemente se equivoca: hueco (falta el balón un momento), salto (la detección se mueve más de lo que vuela un balón), dudoso y perdido. Un clic salta a la escena. Las escenas marcadas pasan a la lista de imágenes como candidatas, con pocas imágenes cada una, y allí revisas cada imagen. Los huecos y los saltos vienen marcados.' },
      ],
      fr: [
        { title: 'Simulation : progression par étapes, annulation avec choix', text: 'La simulation de suivi du ballon affiche désormais trois étapes avec compteur et temps restant : extraire les images, charger le modèle, chercher le ballon. En annulant, vous choisissez de garder les images déjà analysées comme simulation plus courte ou de supprimer toutes les images de l’exécution.' },
        { title: 'Sauter aux scènes notables et les choisir', text: 'Après l’exécution, la simulation liste les scènes où le modèle se trompe probablement : lacune (le ballon manque brièvement), saut (la détection se déplace plus loin que ne vole un ballon), incertain et perdu. Un clic saute à la scène. Les scènes cochées entrent dans la liste d’images comme candidates, avec quelques images chacune, où vous vérifiez chaque image. Les lacunes et les sauts sont cochés au départ.' },
      ],
    },
  },
  {
    version: '0.33.0',
    items: {
      de: [
        { title: 'Lücken aus der Balltracking-Simulation', text: 'Unter dem Player gibt es „Lücken als Prüfkandidaten übernehmen“. Bilder, in denen der Ball zwischen zwei Erkennungen fehlt, werden im Ausschnitt noch einmal durchsucht und kommen als Prüfkandidaten ins Projekt, mit Vorschlag, wenn der Ball dort gefunden wurde. So markierst du genau die Bilder, an denen das Modell scheitert. Videos des unabhängigen Tests bleiben ausgenommen.' },
        { title: 'Reco-Export passt die Sicherheiten an', text: 'Reco nimmt ohne Einstellung jeden Fund ab 10 %; RF-DETR-Modelle melden bei dieser Schwelle viele falsche Bälle und Spieler. Der Export misst jetzt auf den Validierungsbildern die beste Schwelle je Klasse und rechnet die Sicherheiten so um, dass sie in der Datei bei 10 % liegt. Beim Ball zählt die Trefferquote doppelt. Außerdem richtet sich die Datei nach dem häufigsten Bildformat des Projekts und warnt bei gemischten Formaten.' },
      ],
      en: [
        { title: 'Gaps from the ball-tracking simulation', text: 'Below the player there is “Add gaps as review candidates”. Images where the ball is missing between two detections are searched again in a close-up and enter the project as review candidates, with a suggestion when the ball was found there. This way you label exactly the images the model fails on. Videos of the independent test are left out.' },
        { title: 'Reco export fits the confidences', text: 'Without a setting, Reco accepts every detection from 10 %; at that threshold RF-DETR models report many false balls and players. The export now measures the best threshold per class on the validation images and rescales the confidences so that it sits at 10 % in the file. For the ball, finding it counts double. The file also follows the most common image shape of the project and warns about mixed shapes.' },
      ],
      es: [
        { title: 'Huecos de la simulación de seguimiento del balón', text: 'Debajo del reproductor está «Añadir huecos como candidatos». Las imágenes en las que falta el balón entre dos detecciones se vuelven a examinar en un recorte y entran en el proyecto como candidatas, con una propuesta si el balón se encontró allí. Así marcas justo las imágenes en las que falla el modelo. Los vídeos de la prueba independiente quedan fuera.' },
        { title: 'La exportación para Reco ajusta las confianzas', text: 'Sin ajuste, Reco acepta toda detección a partir del 10 %; con ese umbral los modelos RF-DETR dan muchos balones y jugadores falsos. La exportación mide ahora en las imágenes de validación el mejor umbral por clase y reescala las confianzas para que quede en el 10 % en el archivo. En el balón, encontrarlo cuenta el doble. Además, el archivo sigue el formato de imagen más frecuente del proyecto y avisa si hay formatos mezclados.' },
      ],
      fr: [
        { title: 'Lacunes de la simulation de suivi du ballon', text: 'Sous le lecteur se trouve « Ajouter les lacunes comme candidates ». Les images où le ballon manque entre deux détections sont examinées à nouveau dans un recadrage et entrent dans le projet comme candidates, avec une proposition si le ballon y a été trouvé. Vous annotez ainsi précisément les images sur lesquelles le modèle échoue. Les vidéos du test indépendant sont exclues.' },
        { title: 'L’export pour Reco ajuste les confiances', text: 'Sans réglage, Reco accepte toute détection à partir de 10 % ; à ce seuil, les modèles RF-DETR signalent beaucoup de faux ballons et joueurs. L’export mesure désormais sur les images de validation le meilleur seuil par classe et recalcule les confiances pour qu’il se trouve à 10 % dans le fichier. Pour le ballon, le trouver compte double. Le fichier suit aussi le format d’image le plus fréquent du projet et avertit en cas de formats mélangés.' },
      ],
    },
  },
  {
    version: '0.32.0',
    items: {
      de: [{ title: 'Für Reco exportieren', text: 'Ein neuer Knopf neben „Lokal trainieren“ speichert ein Modell aus der Bibliothek als ONNX-Datei, die die Reco-Kamera-App direkt lädt – auch kombinierte Modelle. Voreingestellt ist nur der Ball; Spieler heißen in der Datei „person“, wie Reco sie sucht. Danach prüft ein Probelauf die Datei auf Bildern des Projekts. Der bisherige ONNX-Export hatte ein Format, mit dem Reco nichts fand.' }],
      en: [{ title: 'Export for Reco', text: 'A new button next to “Train locally” saves a library model as an ONNX file the Reco camera app loads directly, combined models included. Only the ball is preselected; players are named “person” in the file, the way Reco looks for them. A test run then checks the file on images of the project. The previous ONNX export had a layout in which Reco found nothing.' }],
      es: [{ title: 'Exportar para Reco', text: 'Un botón nuevo junto a «Entrenar localmente» guarda un modelo de la biblioteca como archivo ONNX que la app de cámara Reco carga directamente, también modelos combinados. Solo el balón viene marcado; los jugadores se llaman «person» en el archivo, como los busca Reco. Después, una prueba comprueba el archivo con imágenes del proyecto. La exportación ONNX anterior tenía un formato en el que Reco no encontraba nada.' }],
      fr: [{ title: 'Exporter pour Reco', text: 'Un nouveau bouton à côté de « Entraîner localement » enregistre un modèle de la bibliothèque en fichier ONNX que l’app caméra Reco charge directement, modèles combinés compris. Seul le ballon est présélectionné ; les joueurs s’appellent « person » dans le fichier, comme Reco les cherche. Un essai vérifie ensuite le fichier sur des images du projet. L’ancien export ONNX avait un format dans lequel Reco ne trouvait rien.' }],
    },
  },
  {
    version: '0.31.8',
    items: {
      de: [
        { title: 'Aktualisieren behält Bilder gelöschter Videos', text: '„Frames sicher aktualisieren“ behält jetzt Bilder und Markierungen von Videos, die nicht mehr im Arbeitsordner liegen. Videos lassen sich also nach dem Extrahieren auf eine externe Platte verschieben. Bereits vorbereitete Videos werden nicht neu extrahiert, nur neue.' },
        { title: 'Training, Validierung und Test sichtbar', text: 'Über der Bildliste zeigen Chips, wie viele Bilder trainieren, validieren und testen; ein Klick filtert die Liste. Jedes Bild zeigt seine Rolle, und der Trainingsdialog warnt bei fehlenden Testbildern oder sehr wenigen Validierungsbildern.' },
        { title: 'Import übernimmt Rollen aus Reco-Datensätzen', text: 'Exportierte Datensätze und Kopien des Projektordners „dataset“ behalten beim Import ihre Aufteilung in Training, Validierung und Test. Ein erneuter Import setzt die Rollen auch bei schon importierten Bildern.' },
      ],
      en: [
        { title: 'Refreshing keeps images of removed videos', text: '“Safely refresh frames” now keeps the images and boxes of videos that are no longer in the working folder, so videos can move to an external drive once extracted. Videos prepared before are not extracted again, only new ones.' },
        { title: 'Training, validation and test at a glance', text: 'Chips above the image list show how many images train, validate and test; a click filters the list. Every image shows its role, and the training dialog warns when test images are missing or validation images are very few.' },
        { title: 'Import keeps the roles of Reco datasets', text: 'Exported datasets and copies of a project’s “dataset” folder keep their training, validation and test roles on import. Importing again sets the roles of images imported before.' },
      ],
      es: [
        { title: 'Actualizar conserva las imágenes de vídeos eliminados', text: '«Actualizar fotogramas de forma segura» conserva ahora las imágenes y marcas de vídeos que ya no están en la carpeta de trabajo, así que los vídeos pueden pasar a un disco externo tras extraerlos. Los vídeos ya preparados no se vuelven a extraer, solo los nuevos.' },
        { title: 'Entrenamiento, validación y prueba a la vista', text: 'Unas etiquetas sobre la lista de imágenes muestran cuántas imágenes entrenan, validan y prueban; un clic filtra la lista. Cada imagen muestra su rol y el diálogo de entrenamiento avisa si faltan imágenes de prueba o hay muy pocas de validación.' },
        { title: 'La importación conserva los roles de conjuntos de Reco', text: 'Los conjuntos exportados y las copias de la carpeta «dataset» del proyecto conservan sus roles de entrenamiento, validación y prueba al importarlos. Importar de nuevo asigna los roles también a imágenes ya importadas.' },
      ],
      fr: [
        { title: 'L’actualisation garde les images des vidéos retirées', text: '« Actualiser les images en sécurité » conserve désormais les images et annotations des vidéos qui ne sont plus dans le dossier de travail : les vidéos peuvent donc passer sur un disque externe une fois extraites. Les vidéos déjà préparées ne sont pas réextraites, seules les nouvelles le sont.' },
        { title: 'Entraînement, validation et test d’un coup d’œil', text: 'Des pastilles au-dessus de la liste d’images indiquent combien d’images servent à l’entraînement, à la validation et au test ; un clic filtre la liste. Chaque image affiche son rôle, et la boîte d’entraînement avertit s’il manque des images de test ou s’il y a très peu d’images de validation.' },
        { title: 'L’import conserve les rôles des jeux Reco', text: 'Les jeux exportés et les copies du dossier « dataset » d’un projet gardent leurs rôles d’entraînement, de validation et de test à l’import. Un nouvel import attribue aussi les rôles aux images déjà importées.' },
      ],
    },
  },
  {
    version: '0.31.7',
    items: {
      de: [{ title: 'Batch-Training läuft nach der Confidence-Prüfung weiter', text: 'Auf einer AMD-Karte unter Windows blieb das Batch-Training nach „Confidence prüfen“ des ersten Modells stehen. Der Batch wartet jetzt nicht mehr endlos: Ein Schritt, der sein Ergebnis geliefert hat, aber nicht endet, wird beendet; eine Confidence-Prüfung, die zehn Minuten lang nichts meldet, wird übersprungen. Das trainierte Modell ist zu dem Zeitpunkt schon gespeichert.' }],
      en: [{ title: 'Batch training continues after the confidence check', text: 'On an AMD card on Windows, batch training stopped after the first model’s “Check confidence”. The batch no longer waits forever: a step that has delivered its result but does not end is stopped, and a confidence check that reports nothing for ten minutes is skipped. The trained model is already saved at that point.' }],
      es: [{ title: 'El entrenamiento por lotes continúa tras comprobar la confianza', text: 'En una tarjeta AMD con Windows, el entrenamiento por lotes se detenía tras «Comprobar la confianza» del primer modelo. El lote ya no espera indefinidamente: un paso que ya entregó su resultado pero no termina se detiene, y una comprobación de confianza que no informa nada durante diez minutos se omite. El modelo entrenado ya está guardado en ese momento.' }],
      fr: [{ title: 'L’entraînement par lots continue après la vérification de la confiance', text: 'Sur une carte AMD sous Windows, l’entraînement par lots s’arrêtait après « Vérifier la confiance » du premier modèle. Le lot n’attend plus indéfiniment : une étape qui a livré son résultat mais ne se termine pas est arrêtée, et une vérification de la confiance qui ne signale rien pendant dix minutes est ignorée. Le modèle entraîné est déjà enregistré à ce moment-là.' }],
    },
  },
  {
    version: '0.31.6',
    items: {
      de: [{ title: 'Batch-Training auf der AMD-Karte bleibt nicht mehr hängen', text: 'Ein Batch-Training auf der AMD-Karte konnte nach einem fertigen Modell stehen bleiben, bevor das nächste begann. Prozesse in der AMD-Umgebung beenden sich jetzt sofort, sobald alles gespeichert ist. Außerdem zeigt der Batch den Start jedes Modells gleich an.' }],
      en: [{ title: 'Batch training on the AMD card no longer gets stuck', text: 'A batch training on the AMD card could stop after a finished model, before the next one began. Processes in the AMD environment now end right away once everything is saved. The batch also shows each model’s start immediately.' }],
      es: [{ title: 'El entrenamiento por lotes en la tarjeta AMD ya no se queda bloqueado', text: 'Un entrenamiento por lotes en la tarjeta AMD podía detenerse tras un modelo terminado, antes de empezar el siguiente. Los procesos del entorno AMD ahora terminan en cuanto todo está guardado. Además, el lote muestra enseguida el inicio de cada modelo.' }],
      fr: [{ title: 'L’entraînement par lots sur la carte AMD ne se bloque plus', text: 'Un entraînement par lots sur la carte AMD pouvait s’arrêter après un modèle terminé, avant le suivant. Les processus de l’environnement AMD se terminent désormais dès que tout est enregistré. Le lot affiche aussi tout de suite le démarrage de chaque modèle.' }],
    },
  },
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
