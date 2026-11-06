import { useEffect, useState } from 'react';
import { hintText } from './hints';

type Language = 'de' | 'en' | 'es' | 'fr';

export type GpuCard = { name: string; vendor: string; vramGB: number };
export type GpuStep = { key: string; title: string; status: 'ok' | 'failed' | 'warning' | 'skipped'; summary: string; error: string | null };
export type GpuInfo = {
  offer: { card: GpuCard; choice: 'yes' | 'no' | null } | null;
  problem: { id: string; reason: string; operation: string; at: number } | null;
  diagnosis: { id: string; ok: boolean; reportPath: string; trigger?: string | null; steps: GpuStep[] } | null;
};

const text = {
  de: {
    askTitle: 'Keine NVIDIA-GPU gefunden',
    ask: (card: GpuCard) => `Auf diesem Windows-PC steht kein CUDA zur Verfügung. Reco Trainer kann deine Grafikkarte ${card.name}${card.vramGB ? ` (${card.vramGB} GB)` : ''} experimentell nutzen: Erkennung über DirectML, VLM über Vulkan. Das Training läuft weiterhin auf der CPU.`,
    askNote: 'Bei „Ja“ wird die Grafikkarte einmal getestet (falls nötig wird dafür zuerst ML eingerichtet). Klappt etwas nicht, erscheint ein Bericht, den du an den Entwickler schicken kannst.',
    yes: 'Ja, Grafikkarte testen', no: 'Nein, nur CPU',
    okTitle: 'Grafikkarte funktioniert', failTitle: 'Grafikkarte funktioniert noch nicht',
    problem: (operation: string, reason: string) => `Bei „${operation}“ ist die Grafikkarte ausgefallen, Reco Trainer hat auf der CPU weitergearbeitet: ${reason}`,
    okIntro: 'Alle Prüfungen waren erfolgreich. Erkennung und VLM nutzen jetzt die Grafikkarte.',
    failIntro: 'Reco Trainer arbeitet auf der CPU weiter, es geht also nichts verloren. Bitte schick den Bericht unten an den Entwickler – er enthält nur Hardware-, Treiber-, Paket- und Log-Angaben, keine Bilder, Videos oder Labels.',
    report: 'Vollständiger Bericht', savedAt: 'Gespeichert unter', copy: 'Bericht kopieren', copied: 'Kopiert ✓', download: 'Als Datei speichern',
    retry: 'Erneut testen', useCpu: 'Nur noch CPU verwenden', close: 'Schließen', loading: 'Bericht wird geladen …',
    status: { ok: 'OK', failed: 'Fehler', warning: 'Hinweis', skipped: 'übersprungen' },
    test: 'Grafikkarte testen (AMD/Intel)',
  },
  en: {
    askTitle: 'No NVIDIA GPU found',
    ask: (card: GpuCard) => `CUDA is not available on this Windows PC. Reco Trainer can use your graphics card ${card.name}${card.vramGB ? ` (${card.vramGB} GB)` : ''} experimentally: detection through DirectML, the VLM through Vulkan. Training still runs on the CPU.`,
    askNote: 'With “Yes”, the graphics card is tested once (ML is set up first if needed). If something fails, a report appears that you can send to the developer.',
    yes: 'Yes, test the graphics card', no: 'No, CPU only',
    okTitle: 'The graphics card works', failTitle: 'The graphics card does not work yet',
    problem: (operation: string, reason: string) => `During “${operation}” the graphics card failed and Reco Trainer continued on the CPU: ${reason}`,
    okIntro: 'All checks passed. Detection and the VLM now use the graphics card.',
    failIntro: 'Reco Trainer continues on the CPU, so nothing is lost. Please send the report below to the developer – it contains hardware, driver, package and log details only, no images, videos or labels.',
    report: 'Full report', savedAt: 'Saved at', copy: 'Copy report', copied: 'Copied ✓', download: 'Save as file',
    retry: 'Test again', useCpu: 'Use the CPU only', close: 'Close', loading: 'Loading the report …',
    status: { ok: 'OK', failed: 'Failed', warning: 'Note', skipped: 'skipped' },
    test: 'Test graphics card (AMD/Intel)',
  },
  es: {
    askTitle: 'No se encontró ninguna GPU NVIDIA',
    ask: (card: GpuCard) => `CUDA no está disponible en este PC con Windows. Reco Trainer puede usar tu gráfica ${card.name}${card.vramGB ? ` (${card.vramGB} GB)` : ''} de forma experimental: detección con DirectML y VLM con Vulkan. El entrenamiento sigue en la CPU.`,
    askNote: 'Con «Sí», la gráfica se prueba una vez (si hace falta, primero se configura ML). Si algo falla, aparece un informe que puedes enviar al desarrollador.',
    yes: 'Sí, probar la gráfica', no: 'No, solo CPU',
    okTitle: 'La gráfica funciona', failTitle: 'La gráfica aún no funciona',
    problem: (operation: string, reason: string) => `Durante «${operation}» la gráfica falló y Reco Trainer continuó en la CPU: ${reason}`,
    okIntro: 'Todas las comprobaciones han ido bien. La detección y el VLM usan ahora la gráfica.',
    failIntro: 'Reco Trainer sigue en la CPU, así que no se pierde nada. Envía el informe de abajo al desarrollador: solo contiene datos de hardware, controladores, paquetes y registros, sin imágenes, vídeos ni etiquetas.',
    report: 'Informe completo', savedAt: 'Guardado en', copy: 'Copiar informe', copied: 'Copiado ✓', download: 'Guardar como archivo',
    retry: 'Probar de nuevo', useCpu: 'Usar solo la CPU', close: 'Cerrar', loading: 'Cargando el informe …',
    status: { ok: 'OK', failed: 'Error', warning: 'Aviso', skipped: 'omitido' },
    test: 'Probar gráfica (AMD/Intel)',
  },
  fr: {
    askTitle: 'Aucune GPU NVIDIA trouvée',
    ask: (card: GpuCard) => `CUDA n’est pas disponible sur ce PC Windows. Reco Trainer peut utiliser votre carte graphique ${card.name}${card.vramGB ? ` (${card.vramGB} Go)` : ''} à titre expérimental : détection via DirectML, VLM via Vulkan. L’entraînement reste sur le CPU.`,
    askNote: 'Avec « Oui », la carte graphique est testée une fois (le ML est configuré d’abord si nécessaire). En cas d’échec, un rapport s’affiche que vous pouvez envoyer au développeur.',
    yes: 'Oui, tester la carte graphique', no: 'Non, CPU uniquement',
    okTitle: 'La carte graphique fonctionne', failTitle: 'La carte graphique ne fonctionne pas encore',
    problem: (operation: string, reason: string) => `Pendant « ${operation} », la carte graphique a échoué et Reco Trainer a continué sur le CPU : ${reason}`,
    okIntro: 'Toutes les vérifications ont réussi. La détection et le VLM utilisent maintenant la carte graphique.',
    failIntro: 'Reco Trainer continue sur le CPU, rien n’est donc perdu. Envoyez le rapport ci-dessous au développeur : il ne contient que des informations sur le matériel, les pilotes, les paquets et les journaux, aucune image, vidéo ni annotation.',
    report: 'Rapport complet', savedAt: 'Enregistré sous', copy: 'Copier le rapport', copied: 'Copié ✓', download: 'Enregistrer comme fichier',
    retry: 'Tester à nouveau', useCpu: 'Utiliser uniquement le CPU', close: 'Fermer', loading: 'Chargement du rapport …',
    status: { ok: 'OK', failed: 'Échec', warning: 'Remarque', skipped: 'ignoré' },
    test: 'Tester la carte graphique (AMD/Intel)',
  },
} as const;

export const gpuText = text;

const SEEN_KEY = 'reco-gpu-report-seen-v1';

function seenReport(): string | null {
  try { return window.localStorage.getItem(SEEN_KEY); } catch { return null; }
}

/** "Try the AMD card?" once per computer, then the report popup after every test or failure. */
export default function GpuDialogs({ api, language, gpu, ready, busy, operationName, choose }: {
  api: string; language: Language; gpu: GpuInfo | undefined;
  /** A working folder is open (the test may need to set up ML in it). */
  ready: boolean; busy: boolean;
  operationName: (operation: string) => string;
  choose: (choice: 'yes' | 'no') => Promise<unknown>;
}) {
  const t = text[language];
  const h = hintText[language];
  const [askDismissed, setAskDismissed] = useState(false);
  const [seen, setSeen] = useState<string | null>(() => seenReport());
  const [report, setReport] = useState<{ id: string; text: string; path: string } | null>(null);
  const [copied, setCopied] = useState(false);
  const diagnosis = gpu?.diagnosis ?? null;
  const showReport = !!diagnosis && diagnosis.id !== seen;

  useEffect(() => {
    if (!showReport || !diagnosis || report?.id === diagnosis.id) return;
    let live = true;
    void fetch(`${api}/api/gpu-report`, { cache: 'no-store' }).then((response) => response.json()).then((value: { id?: string; text?: string; path?: string }) => {
      if (live && value.text) setReport({ id: value.id ?? diagnosis.id, text: value.text, path: value.path ?? '' });
    }).catch(() => undefined);
    return () => { live = false; };
  }, [api, showReport, diagnosis?.id]);

  const markSeen = () => {
    if (!diagnosis) return;
    try { window.localStorage.setItem(SEEN_KEY, diagnosis.id); } catch { /* the popup simply shows again next session */ }
    setSeen(diagnosis.id); setCopied(false);
  };

  if (showReport && diagnosis) {
    const problem = diagnosis.trigger === 'problem' ? gpu?.problem : null;
    const reportText = report?.id === diagnosis.id ? report.text : null;
    const copy = async () => {
      if (!reportText) return;
      try { await navigator.clipboard.writeText(reportText); setCopied(true); }
      catch {
        const area = document.getElementById('gpu-report-text') as HTMLTextAreaElement | null;
        area?.select(); document.execCommand('copy'); setCopied(true);
      }
    };
    const download = () => {
      if (!reportText) return;
      const url = URL.createObjectURL(new Blob([reportText], { type: 'text/plain;charset=utf-8' }));
      const link = document.createElement('a');
      link.href = url; link.download = `reco-trainer-gpu-diagnosis-${diagnosis.id}.txt`;
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
    return <div className="walkthrough-backdrop" role="presentation">
      <section className="onnx-dialog export-dialog gpu-dialog" role="dialog" aria-modal="true" aria-labelledby="gpu-report-title">
        <h2 id="gpu-report-title">{diagnosis.ok ? t.okTitle : t.failTitle}</h2>
        {problem && <p className="gpu-problem">{t.problem(operationName(problem.operation), problem.reason)}</p>}
        <p>{diagnosis.ok && !problem ? t.okIntro : t.failIntro}</p>
        <ul className="gpu-steps">
          {diagnosis.steps.map((step) => <li key={step.key} className={`gpu-step ${step.status}`}>
            <span className="gpu-step-status">{t.status[step.status] ?? step.status}</span>
            <span><strong>{step.title}</strong>{step.summary ? `: ${step.summary}` : ''}{step.error && <small>{step.error}</small>}</span>
          </li>)}
        </ul>
        <details className="gpu-report" open={!diagnosis.ok}>
          <summary>{t.report}</summary>
          {reportText ? <textarea id="gpu-report-text" readOnly value={reportText} spellCheck={false} /> : <small>{t.loading}</small>}
          {report?.path && <small>{t.savedAt}: <code>{report.path}</code></small>}
        </details>
        <div className="onnx-actions gpu-actions">
          <button data-hint={h.gpuCopy} type="button" className="secondary-button" onClick={() => void copy()} disabled={!reportText}>{copied ? t.copied : t.copy}</button>
          <button data-hint={h.gpuDownload} type="button" className="secondary-button" onClick={download} disabled={!reportText}>{t.download}</button>
          {!diagnosis.ok && <button data-hint={h.gpuRetry} type="button" className="secondary-button" onClick={() => { markSeen(); void choose('yes'); }} disabled={busy}>{t.retry}</button>}
          {!diagnosis.ok && gpu?.offer?.choice === 'yes' && <button data-hint={h.gpuUseCpu} type="button" className="secondary-button" onClick={() => { markSeen(); void choose('no'); }}>{t.useCpu}</button>}
          <button data-hint={h.gpuClose} type="button" className="primary-button" onClick={markSeen}>{t.close}</button>
        </div>
      </section>
    </div>;
  }

  const offer = gpu?.offer;
  if (offer && offer.choice === null && ready && !busy && !askDismissed) {
    return <div className="walkthrough-backdrop" role="presentation">
      <section className="onnx-dialog export-dialog gpu-dialog" role="dialog" aria-modal="true" aria-labelledby="gpu-ask-title">
        <h2 id="gpu-ask-title">{t.askTitle}</h2>
        <p>{t.ask(offer.card)}</p>
        <p><small>{t.askNote}</small></p>
        <div className="onnx-actions">
          <button data-hint={h.gpuNo} type="button" className="secondary-button" onClick={() => { setAskDismissed(true); void choose('no'); }}>{t.no}</button>
          <button data-hint={h.gpuYes} type="button" className="primary-button" onClick={() => { setAskDismissed(true); void choose('yes'); }}>{t.yes}</button>
        </div>
      </section>
    </div>;
  }
  return null;
}
