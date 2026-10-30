type Language = 'de' | 'en' | 'es' | 'fr';
type Tier = 'strong' | 'medium' | 'weak';
export type AdvancedSettings = { tier: 'auto' | Tier; teacher: 'auto' | 'nano' | 'small' | 'medium' | 'large'; refineMethod: 'opencv' | 'sam2' };
export type TierInfo = { detected: Tier; plans: Record<Tier, { teacher: string; refiner: string | null; second: string; vlm: boolean }> };
type RegistryModel = { id: string; name: string; purpose?: string; license: string; licenseUrl?: string; bytes?: number; managedBy?: string; installed?: boolean | null };

export const DEFAULT_ADVANCED: AdvancedSettings = { tier: 'auto', teacher: 'auto', refineMethod: 'opencv' };

const text: Record<Language, {
  title: string; tier: string; tierAuto: (tier: string) => string; tiers: Record<Tier, string>; tierHint: string;
  teacher: string; teacherAuto: (size: string) => string; teacherHint: string;
  refine: string; opencv: string; sam2: string; refineHint: string;
  models: string; installed: string; onDemand: (size: string) => string; viaRfdetr: string; purposes: Record<string, string>;
}> = {
  de: {
    title: 'Erweiterte Optionen', tier: 'Hardware-Stufe', tierAuto: (tier) => `Automatisch (${tier})`,
    tiers: { strong: 'stark', medium: 'mittel', weak: 'schwach' },
    tierHint: 'Bestimmt, welche Modelle genutzt werden. Stark: NVIDIA mit 16 GB oder Apple mit 32 GB und mehr. Schwach: nur CPU – dann ohne SAM 2.',
    teacher: 'Vorschlagsmodell (solange kein eigenes Modell die Klasse kennt)', teacherAuto: (size) => `Automatisch (RF-DETR ${size})`,
    teacherHint: 'Größere Modelle finden mehr, brauchen aber mehr Zeit und Speicher.',
    refine: 'Boxen verfeinern mit', opencv: 'OpenCV', sam2: 'SAM 2 (experimentell)',
    refineHint: 'SAM 2 übernimmt nur kleine Korrekturen. In einem ersten Vergleich war es nicht messbar besser als OpenCV – bitte Stichproben prüfen.',
    models: 'Modelle und Lizenzen', installed: 'geladen', onDemand: (size) => `wird bei Bedarf geladen (${size})`, viaRfdetr: 'über RF-DETR',
    purposes: { detector: 'Erkennung', 'anonymize-faces': 'Anonymisierung', 'anonymize-text': 'Text unkenntlich', 'refine-boxes': 'Boxen verfeinern', 'second-teacher': 'Zweite Meinung' },
  },
  en: {
    title: 'Advanced options', tier: 'Hardware tier', tierAuto: (tier) => `Automatic (${tier})`,
    tiers: { strong: 'strong', medium: 'medium', weak: 'weak' },
    tierHint: 'Decides which models are used. Strong: NVIDIA with 16 GB or Apple with 32 GB or more. Weak: CPU only – then without SAM 2.',
    teacher: 'Suggestion model (while no trained model knows the class)', teacherAuto: (size) => `Automatic (RF-DETR ${size})`,
    teacherHint: 'Larger models find more but need more time and memory.',
    refine: 'Refine boxes with', opencv: 'OpenCV', sam2: 'SAM 2 (experimental)',
    refineHint: 'SAM 2 only takes over small corrections. In a first comparison it was not measurably better than OpenCV – please check samples.',
    models: 'Models and licenses', installed: 'downloaded', onDemand: (size) => `downloaded when needed (${size})`, viaRfdetr: 'via RF-DETR',
    purposes: { detector: 'detection', 'anonymize-faces': 'anonymization', 'anonymize-text': 'hiding text', 'refine-boxes': 'refining boxes', 'second-teacher': 'second opinion' },
  },
  es: {
    title: 'Opciones avanzadas', tier: 'Nivel de hardware', tierAuto: (tier) => `Automático (${tier})`,
    tiers: { strong: 'alto', medium: 'medio', weak: 'bajo' },
    tierHint: 'Decide qué modelos se usan. Alto: NVIDIA con 16 GB o Apple con 32 GB o más. Bajo: solo CPU, entonces sin SAM 2.',
    teacher: 'Modelo de sugerencias (mientras ningún modelo propio conozca la clase)', teacherAuto: (size) => `Automático (RF-DETR ${size})`,
    teacherHint: 'Los modelos grandes encuentran más, pero necesitan más tiempo y memoria.',
    refine: 'Refinar cuadros con', opencv: 'OpenCV', sam2: 'SAM 2 (experimental)',
    refineHint: 'SAM 2 solo aplica correcciones pequeñas. En una primera comparación no fue mejor que OpenCV de forma medible: revisa algunas muestras.',
    models: 'Modelos y licencias', installed: 'descargado', onDemand: (size) => `se descarga cuando haga falta (${size})`, viaRfdetr: 'mediante RF-DETR',
    purposes: { detector: 'detección', 'anonymize-faces': 'anonimización', 'anonymize-text': 'ocultar texto', 'refine-boxes': 'refinar cuadros', 'second-teacher': 'segunda opinión' },
  },
  fr: {
    title: 'Options avancées', tier: 'Niveau matériel', tierAuto: (tier) => `Automatique (${tier})`,
    tiers: { strong: 'élevé', medium: 'moyen', weak: 'faible' },
    tierHint: 'Détermine les modèles utilisés. Élevé : NVIDIA avec 16 Go ou Apple avec 32 Go ou plus. Faible : CPU uniquement – alors sans SAM 2.',
    teacher: 'Modèle de suggestions (tant qu’aucun modèle entraîné ne connaît la classe)', teacherAuto: (size) => `Automatique (RF-DETR ${size})`,
    teacherHint: 'Les grands modèles trouvent davantage mais demandent plus de temps et de mémoire.',
    refine: 'Affiner les boîtes avec', opencv: 'OpenCV', sam2: 'SAM 2 (expérimental)',
    refineHint: 'SAM 2 n’applique que de petites corrections. Dans une première comparaison, il n’était pas mesurablement meilleur qu’OpenCV – vérifiez des échantillons.',
    models: 'Modèles et licences', installed: 'téléchargé', onDemand: (size) => `téléchargé si nécessaire (${size})`, viaRfdetr: 'via RF-DETR',
    purposes: { detector: 'détection', 'anonymize-faces': 'anonymisation', 'anonymize-text': 'masquer le texte', 'refine-boxes': 'affiner les boîtes', 'second-teacher': 'second avis' },
  },
};

const sizeLabel = (bytes?: number) => !bytes ? '' : bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
const teacherSize = (id: string) => id.replace('rf-detr-', '').replace(/^./, (letter) => letter.toUpperCase());

/** Hardware tier, teacher and refinement choices plus the model/license list (training panel). */
export function AdvancedOptions({ language, tierInfo, registry, settings, onChange }: {
  language: Language; tierInfo?: TierInfo; registry: RegistryModel[]; settings: AdvancedSettings; onChange: (settings: AdvancedSettings) => void;
}) {
  const t = text[language];
  const detected = tierInfo?.detected ?? 'weak';
  const effective: Tier = settings.tier === 'auto' ? detected : settings.tier;
  const plan = tierInfo?.plans?.[effective];
  return <details className="advanced-options">
    <summary>{t.title}</summary>
    <div className="advanced-grid">
    <label data-hint={t.tierHint}>{t.tier}
      <select value={settings.tier} onChange={(event) => onChange({ ...settings, tier: event.target.value as AdvancedSettings['tier'] })}>
        <option value="auto">{t.tierAuto(t.tiers[detected])}</option>
        {(['strong', 'medium', 'weak'] as Tier[]).map((tier) => <option key={tier} value={tier}>{t.tiers[tier]}</option>)}
      </select>
      <small>{t.tierHint}</small>
    </label>
    <label data-hint={t.teacherHint}>{t.teacher}
      <select value={settings.teacher} onChange={(event) => onChange({ ...settings, teacher: event.target.value as AdvancedSettings['teacher'] })}>
        <option value="auto">{t.teacherAuto(plan ? teacherSize(plan.teacher) : 'S')}</option>
        {(['nano', 'small', 'medium', 'large'] as const).map((size) => <option key={size} value={size}>RF-DETR {teacherSize(size)}</option>)}
      </select>
      <small>{t.teacherHint}</small>
    </label>
    <label data-hint={t.refineHint}>{t.refine}
      <select value={settings.refineMethod} onChange={(event) => onChange({ ...settings, refineMethod: event.target.value as AdvancedSettings['refineMethod'] })}>
        <option value="opencv">{t.opencv}</option>
        <option value="sam2">{t.sam2}</option>
      </select>
      <small>{t.refineHint}</small>
    </label>
    <div className="advanced-models"><strong>{t.models}</strong><ul>
      {registry.map((model) => <li key={model.id}>
        <span>{model.name}</span>
        <span className="advanced-model-meta">{t.purposes[model.purpose ?? ''] ?? model.purpose} · {model.licenseUrl ? <a href={model.licenseUrl} target="_blank" rel="noreferrer">{model.license}</a> : model.license} · {model.managedBy ? t.viaRfdetr : model.installed ? `✓ ${t.installed}` : t.onDemand(sizeLabel(model.bytes))}</span>
      </li>)}
    </ul></div>
    </div>
  </details>;
}
