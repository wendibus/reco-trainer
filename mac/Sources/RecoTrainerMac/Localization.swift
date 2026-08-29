import Foundation

struct WalkthroughStep {
    let title: String
    let detail: String
}

enum AppLanguage: String, CaseIterable, Identifiable {
    case de
    case en
    case es
    case fr

    var id: String { rawValue }

    var title: String {
        switch self {
        case .de: "DE"
        case .en: "EN"
        case .es: "ES"
        case .fr: "FR"
        }
    }
}

extension AppLanguage {
    private static let spanish: [String: String] = [
        "Notice": "Aviso", "Language": "Idioma", "1 · Sport": "1 · Deporte", "Sport": "Deporte",
        "Select video folder": "Seleccionar carpeta de vídeos", "No folder selected": "Ninguna carpeta seleccionada",
        "Analyze videos locally": "Analizar vídeos localmente", "3 · Training images": "3 · Imágenes de entrenamiento",
        "No videos or frames are uploaded.": "No se sube ningún vídeo ni fotograma.", "Annotate object:": "Marcar objeto:",
        "Class": "Clase", "4 · Improve model": "4 · Mejorar modelo", "Model": "Modelo",
        "Check hardware": "Comprobar hardware", "Set up ML": "Configurar ML", "Train locally": "Entrenar localmente",
        "CPU model (ONNX)": "Modelo CPU (ONNX)", "Apple model (Core ML)": "Modelo Apple (Core ML)",
        "Exchange package": "Crear paquete", "Import model": "Importar modelo", "Select Reco model package": "Seleccionar paquete de modelo Reco", "Active exchange model": "Modelo de intercambio activo", "Minimum confidence": "Confianza mínima",
        "Local training for Reco Cam": "Entrenamiento local para Reco Cam", "Frame is missing": "Falta el fotograma",
        "Annotate": "Marcar", "Pan": "Mover", "Drag image; pinch to zoom": "Arrastra la imagen; pellizca para ampliar",
        "Draw a box; pinch to zoom": "Dibuja un cuadro; pellizca para ampliar", "Reset": "Restablecer",
        "Remove last": "Eliminar el último", "Clear frame": "Vaciar fotograma", "Checking Mac hardware …": "Comprobando el hardware del Mac …",
        "Setting up local ML environment …": "Configurando el entorno ML local …", "Training locally …": "Entrenando localmente …",
        "Exporting universal CPU model …": "Exportando el modelo universal para CPU …", "Exporting Core ML model …": "Exportando el modelo Core ML …",
        "Creating privacy-safe exchange package …": "Creando un paquete de intercambio privado …", "Validating and importing model package …": "Validando e importando el paquete del modelo …",
        "Model import failed.": "La importación del modelo ha fallado.", "Done.": "Listo.", "Operation failed.": "La operación ha fallado.", "Import model packages only from a trusted source.": "Importa paquetes de modelos solo de una fuente de confianza.",
        "MPS uses the Apple GPU for training. Parallel data loaders keep it supplied.": "MPS usa la GPU de Apple para el entrenamiento. Los cargadores de datos paralelos la mantienen ocupada.",
        "Note: The generic base model only knows “person” and cannot yet reliably distinguish players, referees, and spectators.": "Nota: el modelo base genérico solo conoce «persona» y todavía no distingue de forma fiable jugadores, árbitros y espectadores.",
        "Choose a sport, open a video folder, and correct errors directly on your Mac.\nThe original recordings never leave the computer.": "Elige un deporte, abre una carpeta de vídeos y corrige los errores directamente en tu Mac.\nLas grabaciones originales nunca salen del equipo."
    ]

    private static let french: [String: String] = [
        "Notice": "Information", "Language": "Langue", "1 · Sport": "1 · Sport", "Sport": "Sport",
        "Select video folder": "Sélectionner le dossier vidéo", "No folder selected": "Aucun dossier sélectionné",
        "Analyze videos locally": "Analyser les vidéos localement", "3 · Training images": "3 · Images d’entraînement",
        "No videos or frames are uploaded.": "Aucune vidéo ni image n’est envoyée.", "Annotate object:": "Annoter l’objet :",
        "Class": "Classe", "4 · Improve model": "4 · Améliorer le modèle", "Model": "Modèle",
        "Check hardware": "Vérifier le matériel", "Set up ML": "Configurer le ML", "Train locally": "Entraîner localement",
        "CPU model (ONNX)": "Modèle CPU (ONNX)", "Apple model (Core ML)": "Modèle Apple (Core ML)",
        "Exchange package": "Créer le paquet", "Import model": "Importer un modèle", "Select Reco model package": "Sélectionner un paquet de modèle Reco", "Active exchange model": "Modèle d’échange actif", "Minimum confidence": "Confiance minimale",
        "Local training for Reco Cam": "Entraînement local pour Reco Cam", "Frame is missing": "Image manquante",
        "Annotate": "Annoter", "Pan": "Déplacer", "Drag image; pinch to zoom": "Faites glisser l’image ; pincez pour zoomer",
        "Draw a box; pinch to zoom": "Tracez une boîte ; pincez pour zoomer", "Reset": "Réinitialiser",
        "Remove last": "Supprimer la dernière", "Clear frame": "Vider l’image", "Checking Mac hardware …": "Vérification du matériel Mac …",
        "Setting up local ML environment …": "Configuration de l’environnement ML local …", "Training locally …": "Entraînement local …",
        "Exporting universal CPU model …": "Exportation du modèle CPU universel …", "Exporting Core ML model …": "Exportation du modèle Core ML …",
        "Creating privacy-safe exchange package …": "Création du paquet d’échange confidentiel …", "Validating and importing model package …": "Validation et importation du paquet du modèle …",
        "Model import failed.": "Échec de l’importation du modèle.", "Done.": "Terminé.", "Operation failed.": "Échec de l’opération.", "Import model packages only from a trusted source.": "Importez uniquement des paquets provenant d’une source fiable.",
        "MPS uses the Apple GPU for training. Parallel data loaders keep it supplied.": "MPS utilise le GPU Apple pour l’entraînement. Les chargeurs de données parallèles l’alimentent en continu.",
        "Note: The generic base model only knows “person” and cannot yet reliably distinguish players, referees, and spectators.": "Remarque : le modèle générique ne connaît que « personne » et ne distingue pas encore fiablement joueurs, arbitres et spectateurs.",
        "Choose a sport, open a video folder, and correct errors directly on your Mac.\nThe original recordings never leave the computer.": "Choisissez un sport, ouvrez un dossier vidéo et corrigez les erreurs directement sur votre Mac.\nLes enregistrements originaux ne quittent jamais l’ordinateur."
    ]

    func text(_ german: String, _ english: String, _ explicitSpanish: String? = nil, _ explicitFrench: String? = nil) -> String {
        switch self {
        case .de: german
        case .en: english
        case .es: explicitSpanish ?? Self.spanish[english] ?? english
        case .fr: explicitFrench ?? Self.french[english] ?? english
        }
    }

    func category(_ rawValue: String) -> String {
        switch rawValue {
        case "ball": self == .es ? "Balón" : self == .fr ? "Ballon" : "Ball"
        case "puck": self == .es ? "Disco" : self == .fr ? "Palet" : "Puck"
        case "player": self == .de ? "Spieler" : self == .es ? "Jugador" : self == .fr ? "Joueur" : "Player"
        case "goalkeeper": self == .de ? "Torwart" : self == .es ? "Portero" : self == .fr ? "Gardien" : "Goalkeeper"
        case "referee": self == .de ? "Schiedsrichter" : self == .es ? "Árbitro" : self == .fr ? "Arbitre" : "Referee"
        case "goal": self == .de ? "Tor" : self == .es ? "Portería" : self == .fr ? "But" : "Goal"
        case "goalpost": self == .de ? "Torpfosten" : self == .es ? "Postes" : self == .fr ? "Poteaux" : "Goalpost"
        case "hoop": self == .de ? "Korb" : self == .es ? "Canasta" : self == .fr ? "Panier" : "Hoop"
        default: rawValue
        }
    }

    var walkthroughTitle: String {
        text(
            "So verbesserst du dein Reco-Modell",
            "How to improve your Reco model",
            "Cómo mejorar tu modelo Reco",
            "Comment améliorer votre modèle Reco"
        )
    }

    var walkthroughSteps: [WalkthroughStep] {
        switch self {
        case .de:
            [
                .init(title: "Sportart wählen", detail: "Wähle zuerst die Sportart. Jede Sportart besitzt passende Objektklassen und eine getrennte Modellbibliothek."),
                .init(title: "Lokalen Videoordner verbinden", detail: "Wähle den Ordner mit deinen Videos. Die Aufnahmen bleiben auf diesem Rechner und werden nicht hochgeladen."),
                .init(title: "Trainingsbilder vorbereiten", detail: "Reco Trainer extrahiert lokal einzelne Frames und legt das Trainingsprojekt im gewählten Ordner an."),
                .init(title: "Automatisch markieren und prüfen", detail: "Wähle eine Klasse, starte die automatische Markierung, korrigiere Boxen und lösche falsche Treffer."),
                .init(title: "Lokal trainieren", detail: "Richte ML einmalig ein und trainiere anschließend das sportartspezifische Modell auf deiner lokalen Hardware."),
                .init(title: "Exportieren oder austauschen", detail: "Exportiere ein Gerätemodell oder erstelle ein .recomodel-Paket. Es enthält keine Videos oder Frames."),
                .init(title: "Modelle objektiv vergleichen", detail: "Lege geprüfte Boxen als richtige Antworten fest und teste alle kompatiblen lokalen Modelle auf denselben Bildern."),
            ]
        case .en:
            [
                .init(title: "Choose a sport", detail: "Choose the sport first. Each sport has suitable object classes and a separate model library."),
                .init(title: "Connect a local video folder", detail: "Select the folder containing your videos. Recordings stay on this computer and are never uploaded."),
                .init(title: "Prepare training images", detail: "Reco Trainer extracts individual frames locally and creates the training project inside the selected folder."),
                .init(title: "Auto-label and review", detail: "Choose a class, run auto-labeling, correct boxes, and delete false detections."),
                .init(title: "Train locally", detail: "Set up ML once, then train the sport-specific model on your local hardware."),
                .init(title: "Export or exchange", detail: "Export a device model or create a .recomodel package. It contains no videos or frames."),
                .init(title: "Compare models objectively", detail: "Freeze reviewed boxes as the correct answers and test every compatible local model on the same images."),
            ]
        case .es:
            [
                .init(title: "Elige un deporte", detail: "Elige primero el deporte. Cada deporte tiene clases de objetos adecuadas y una biblioteca de modelos independiente."),
                .init(title: "Conecta una carpeta local", detail: "Selecciona la carpeta que contiene tus vídeos. Las grabaciones permanecen en este equipo y nunca se suben."),
                .init(title: "Prepara las imágenes", detail: "Reco Trainer extrae fotogramas localmente y crea el proyecto dentro de la carpeta seleccionada."),
                .init(title: "Marca automáticamente y revisa", detail: "Elige una clase, ejecuta el marcado automático, corrige cuadros y elimina detecciones falsas."),
                .init(title: "Entrena localmente", detail: "Configura ML una vez y entrena el modelo específico del deporte con tu hardware local."),
                .init(title: "Exporta o intercambia", detail: "Exporta un modelo o crea un paquete .recomodel. No contiene vídeos ni fotogramas."),
                .init(title: "Compara modelos objetivamente", detail: "Fija los cuadros revisados como respuestas correctas y prueba todos los modelos locales compatibles con las mismas imágenes."),
            ]
        case .fr:
            [
                .init(title: "Choisissez un sport", detail: "Choisissez d’abord le sport. Chaque sport possède des classes adaptées et une bibliothèque de modèles distincte."),
                .init(title: "Connectez un dossier local", detail: "Sélectionnez le dossier contenant vos vidéos. Les enregistrements restent sur cet ordinateur et ne sont jamais envoyés."),
                .init(title: "Préparez les images", detail: "Reco Trainer extrait localement des images et crée le projet dans le dossier sélectionné."),
                .init(title: "Marquez automatiquement et vérifiez", detail: "Choisissez une classe, lancez le marquage automatique, corrigez les boîtes et supprimez les détections erronées."),
                .init(title: "Entraînez localement", detail: "Configurez le ML une fois, puis entraînez le modèle propre au sport sur votre matériel local."),
                .init(title: "Exportez ou échangez", detail: "Exportez un modèle ou créez un paquet .recomodel. Il ne contient ni vidéos ni images."),
                .init(title: "Comparez objectivement les modèles", detail: "Figez les boîtes vérifiées comme bonnes réponses et testez tous les modèles locaux compatibles sur les mêmes images."),
            ]
        }
    }
}
