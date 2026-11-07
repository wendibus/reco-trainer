import { useEffect } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';
type Item = { title: string; text: string };

/** Shown once after an update (see page.tsx, key reco-whats-new-seen-v1); clicking the version
 *  number in the header opens it again. Add the next version's entry at the top. */
export const WHATS_NEW: { version: string; items: Record<Language, Item[]> }[] = [
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
