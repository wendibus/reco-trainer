/** Texts for training on an AMD card with ROCm (Windows), see amd_training.py. */
type Language = 'de' | 'en' | 'es' | 'fr';

export type RocmStatus = { card?: string | null; installed: boolean; ready: boolean; enabled: boolean; device?: string | null; problem?: { message: string; at: string } | null };

export const rocmText: Record<Language, { setup: string; setupHint: string; confirm: (card: string) => string; toggle: string; toggleHint: string; accelerator: (name?: string | null) => string }> = {
  de: {
    setup: 'AMD-Training einrichten',
    setupHint: 'Installiert AMDs PyTorch mit ROCm in einen eigenen Ordner, damit „Lokal trainieren“ die Grafikkarte nutzt. Braucht Python 3.12 und etwa 6 GB.',
    confirm: (card) => `AMD-Training für ${card} einrichten?\n\nDas lädt etwa 4 GB von AMD und PyTorch und braucht rund 6 GB Platz. Python 3.12 muss installiert sein. Deine bisherige ML-Umgebung bleibt unverändert.`,
    toggle: 'Auf der AMD-Karte trainieren',
    toggleHint: 'An: „Lokal trainieren“ (RF-DETR) läuft über ROCm auf der Grafikkarte. Aus: Training auf der CPU. Erkennung und YOLO laufen immer wie bisher.',
    accelerator: (name) => `Training auf ${name || 'der AMD-Karte'} (ROCm) · Erkennung über DirectML`,
  },
  en: {
    setup: 'Set up AMD training',
    setupHint: 'Installs AMD’s PyTorch with ROCm into a separate folder so “Train locally” uses the graphics card. Needs Python 3.12 and about 6 GB.',
    confirm: (card) => `Set up AMD training for ${card}?\n\nThis downloads about 4 GB from AMD and PyTorch and needs about 6 GB of space. Python 3.12 must be installed. Your current ML environment stays as it is.`,
    toggle: 'Train on the AMD card',
    toggleHint: 'On: “Train locally” (RF-DETR) runs on the graphics card through ROCm. Off: training on the CPU. Detection and YOLO always run as before.',
    accelerator: (name) => `Training on ${name || 'the AMD card'} (ROCm) · detection through DirectML`,
  },
  es: {
    setup: 'Configurar entrenamiento AMD',
    setupHint: 'Instala PyTorch con ROCm de AMD en una carpeta aparte para que «Entrenar localmente» use la gráfica. Necesita Python 3.12 y unos 6 GB.',
    confirm: (card) => `¿Configurar el entrenamiento AMD para ${card}?\n\nSe descargan unos 4 GB de AMD y PyTorch y se necesitan unos 6 GB de espacio. Python 3.12 debe estar instalado. Tu entorno ML actual no cambia.`,
    toggle: 'Entrenar en la tarjeta AMD',
    toggleHint: 'Activado: «Entrenar localmente» (RF-DETR) usa la gráfica mediante ROCm. Desactivado: entrenamiento en la CPU. La detección y YOLO funcionan siempre como antes.',
    accelerator: (name) => `Entrenamiento en ${name || 'la tarjeta AMD'} (ROCm) · detección con DirectML`,
  },
  fr: {
    setup: 'Configurer l’entraînement AMD',
    setupHint: 'Installe PyTorch avec ROCm d’AMD dans un dossier séparé pour que « Entraîner localement » utilise la carte graphique. Nécessite Python 3.12 et environ 6 Go.',
    confirm: (card) => `Configurer l’entraînement AMD pour ${card} ?\n\nCela télécharge environ 4 Go depuis AMD et PyTorch et nécessite environ 6 Go d’espace. Python 3.12 doit être installé. Votre environnement ML actuel reste inchangé.`,
    toggle: 'Entraîner sur la carte AMD',
    toggleHint: 'Activé : « Entraîner localement » (RF-DETR) utilise la carte graphique via ROCm. Désactivé : entraînement sur le CPU. La détection et YOLO fonctionnent toujours comme avant.',
    accelerator: (name) => `Entraînement sur ${name || 'la carte AMD'} (ROCm) · détection via DirectML`,
  },
};
