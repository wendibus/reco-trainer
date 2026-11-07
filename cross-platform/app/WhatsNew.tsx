import { useEffect } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type Item = { title: string; text: string };

/** Shown once after an update (see page.tsx, key reco-whats-new-seen-v1); clicking the version
 *  number in the header opens it again. Add the next version's entry at the top. */
export const WHATS_NEW: { version: string; items: Record<Language, Item[]> }[] = [
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
