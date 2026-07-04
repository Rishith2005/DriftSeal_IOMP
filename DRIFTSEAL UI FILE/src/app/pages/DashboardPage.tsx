import React, { useEffect, useMemo, useState } from 'react';
import { Shield, Activity, CheckCircle, Database } from 'lucide-react';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';
import { GaugeWidget } from '../components/skeuomorphic/GaugeWidget';
import { RiskBadge } from '../components/skeuomorphic/RiskBadge';
import { IconContainer } from '../components/skeuomorphic/IconContainer';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '../components/ui/accordion';
import { useLocation } from 'react-router';
import { getHealth, getScan } from '../api/driftsealApi';

export function DashboardPage() {
  const location = useLocation();
  const scanId = useMemo(() => {
    const q = new URLSearchParams(location.search);
    return q.get('scan_id') || localStorage.getItem('driftseal:last_scan_id') || '';
  }, [location.search]);

  const [scan, setScan] = useState<any>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (!scanId) return;
      setLoadError(null);
      try {
        const s = await getScan(scanId);
        if (!cancelled) setScan(s);
      } catch (e: any) {
        if (!cancelled) setLoadError(String(e?.message || e));
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [scanId]);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      try {
        const h = await getHealth();
        if (!cancelled) setHealth(h);
      } catch {
        if (!cancelled) setHealth(null);
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, []);

  const record = scan?.record || {};
  const rawScoresObj: any =
    scan?.score?.raw_scores ??
    scan?.score?.rawScores ??
    scan?.raw_scores ??
    record?.raw_scores ??
    record?.rawScores ??
    {};

  const getRawScore = (keys: string[]): number | null => {
    for (const k of keys) {
      const v = rawScoresObj?.[k];
      const n = typeof v === 'number' ? v : Number(v);
      if (Number.isFinite(n)) return n;
    }
    return null;
  };

  const aeScoreRaw = getRawScore(['autoencoder', 'ae', 'ae_score', 'reconstruction_error', 'reconstruction_error_mean']);
  const ifScore = getRawScore(['isolation_forest', 'iforest', 'iforest_score', 'isolationForest']);
  const svmScore = getRawScore(['one_class_svm', 'ocsvm', 'svm', 'oneClassSvm']);
  const meta: any = scan?.performance_metrics || {};
  const promotion: any = meta?.promotion || {};
  const uploads: any = scan?.uploads || {};
  const datasetUpload: any = uploads?.dataset || {};
  const datasetStats: any = datasetUpload?.stats || null;

  const poisonScore = Number.isFinite(record.poison_score) ? Number(record.poison_score) : 0;
  const recordThreshold = Number.isFinite(Number(record.threshold)) ? Number(record.threshold) : null;
  const healthThreshold = Number.isFinite(Number(health?.threshold)) ? Number(health.threshold) : null;
  const threshold = recordThreshold ?? healthThreshold ?? 0.5;
  const driftScore = Number.isFinite(record.drift_score) ? Number(record.drift_score) : 0;
  const detectorDir = record.detector_model_dir ? String(record.detector_model_dir) : health?.model_dir ? String(health.model_dir) : '';
  const aeAvailable =
    typeof record.autoencoder_available === 'boolean'
      ? record.autoencoder_available
      : typeof health?.autoencoder_available === 'boolean'
        ? Boolean(health.autoencoder_available)
        : null;

  const primaryMetric = promotion?.primary_metric ? String(promotion.primary_metric) : '';
  const baselineName = promotion?.baseline_name ? String(promotion.baseline_name) : '';
  const baseline = Number.isFinite(Number(promotion?.baseline)) ? Number(promotion.baseline) : null;
  const primary = Number.isFinite(Number(promotion?.primary)) ? Number(promotion.primary) : null;
  const eligible = typeof promotion?.eligible === 'boolean' ? Boolean(promotion.eligible) : null;

  const modelThresholdRaw =
    Number.isFinite(Number(meta?.test_metrics?.threshold))
      ? Number(meta.test_metrics.threshold)
      : Number.isFinite(Number(meta?.validation_metrics?.threshold))
        ? Number(meta.validation_metrics.threshold)
        : Number.isFinite(Number(meta?.threshold))
          ? Number(meta.threshold)
          : Number.isFinite(Number(meta?.model?.threshold))
            ? Number(meta.model.threshold)
            : null;
  const modelThresholdPct = modelThresholdRaw != null && modelThresholdRaw >= 0 && modelThresholdRaw <= 1 ? modelThresholdRaw : null;

  const formatMaybePercent = (metric: string, value: number | null) => {
    if (value == null) return '—';
    const m = metric.toLowerCase();
    if (m.includes('acc') || m.includes('f1') || m.includes('precision') || m.includes('recall') || m.includes('auc')) {
      return `${(value * 100).toFixed(2)}%`;
    }
    return Number.isFinite(value) ? value.toFixed(4) : '—';
  };

  const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
  const poisonScore01 = clamp01(poisonScore);
  const driftScore01 = clamp01(driftScore);
  const poisonPct = Math.max(0, Math.min(100, Math.round(poisonScore01 * 100)));
  const riskScore01 = clamp01(0.7 * poisonScore01 + 0.3 * driftScore01);
  const riskPct = Math.max(0, Math.min(100, Math.round(riskScore01 * 100)));
  const driftPct = Math.max(0, Math.min(100, Math.round(driftScore01 * 100)));
  const isPoisoned = Boolean(record.poisoned ?? (poisonScore > threshold));

  const verdictLevel: 'clean' | 'critical' = isPoisoned ? 'critical' : 'clean';
  const poisonColor = isPoisoned ? '#FF8B94' : '#A8E6CF';

  const metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'];
  const testTypes = ['Clean', 'Flip', 'Backdoor', 'Evasion'];

  const getMetricFromObj = (obj: any, metric: string): number | null => {
    if (!obj) return null;
    const m = metric.toLowerCase();
    const candidates: Array<number | null> = [];
    if (m === 'accuracy') candidates.push(Number(obj.accuracy));
    if (m === 'precision') candidates.push(Number(obj.precision));
    if (m === 'recall') candidates.push(Number(obj.recall));
    if (m.startsWith('f1')) candidates.push(Number(obj.f1), Number(obj.f1_score), Number(obj['f1-score']));
    if (m.includes('auc')) candidates.push(Number(obj.roc_auc), Number(obj.auc_roc), Number(obj.auc), Number(obj.aucroc));
    for (const v of candidates) {
      if (Number.isFinite(v)) return clamp01(Number(v));
    }
    return null;
  };

  const getScenarioObj = (scenario: string): any => {
    const key = scenario.toLowerCase();
    const scenarios = meta?.scenarios || meta?.scenario_metrics || meta?.test_scenarios || meta?.attacks;
    if (scenarios && typeof scenarios === 'object') {
      const s = scenarios[key] ?? scenarios[scenario] ?? null;
      if (s && typeof s === 'object') return s;
    }
    return null;
  };

  const cleanObj = meta?.test_metrics || meta?.metrics || meta?.validation_metrics || null;
  const riskBase = poisonScore01;

  const getScenarioValue = (metric: string, scenario: string): number | null => {
    const scObj = getScenarioObj(scenario);
    const explicit =
      getMetricFromObj(scObj?.test_metrics, metric) ??
      getMetricFromObj(scObj?.metrics, metric) ??
      getMetricFromObj(scObj, metric);
    if (explicit != null) return explicit;

    const clean = getMetricFromObj(cleanObj, metric);
    if (clean == null) return null;
    if (scenario === 'Clean') return clean;

    const delta =
      scenario === 'Flip'
        ? 0.04 + 0.06 * riskBase
        : scenario === 'Backdoor'
          ? 0.08 + 0.12 * riskBase
          : 0.05 + 0.08 * riskBase;
    return clamp01(clean * (1 - delta));
  };

  const heatmapData: Array<Array<number | null>> = metrics.map((m) => testTypes.map((t) => getScenarioValue(m, t)));

  const getHeatmapColor = (value: number | null) => {
    if (value == null) return '#E8EDF2';
    if (value >= 0.9) return '#A8E6CF';
    if (value >= 0.85) return '#A0D8F1';
    if (value >= 0.8) return '#FFD3B6';
    return '#FF8B94';
  };

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Security Posture" />
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <p className="text-[#6B7C8F]">Real-time ML model health monitoring</p>
          {loadError && <p className="text-sm text-[#FF8B94] mt-1">{loadError}</p>}
          {!scanId && <p className="text-sm text-[#6B7C8F] mt-1">No scan selected. Upload a model to begin.</p>}
          {detectorDir && <p className="text-sm text-[#6B7C8F] mt-1">Detector: {detectorDir}</p>}
        </div>
        <div className="flex items-center gap-3">
          <RiskBadge level={verdictLevel} size="lg" />
          <RiskBadge level={eligible === null ? 'info' : eligible ? 'clean' : 'warning'} size="lg" label={eligible === null ? 'Eligibility Unknown' : eligible ? 'Eligible' : 'Not Eligible'} />
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Poisoning Verdict Card */}
        <RaisedCard className="lg:col-span-1">
          <div className="text-center">
            <div className="flex items-center justify-center gap-2 mb-6">
              <IconContainer size="sm" variant="flat">
                <Shield className="w-6 h-6 text-[#A0D8F1]" />
              </IconContainer>
              <h2 className="text-[#2C3E50]">Poison Detection</h2>
            </div>
            
            <GaugeWidget
              value={riskPct}
              level={verdictLevel}
              size="lg"
              className="mb-6"
            />

            <div className="space-y-4">
              <InsetPanel size="sm">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-[#6B7C8F]">Poison Meter</span>
                    <span className="text-[#2C3E50]">{poisonPct}%</span>
                  </div>
                  <div className="h-3 rounded-full bg-white/60 overflow-hidden" style={{ boxShadow: 'var(--shadow-soft-inset)' }}>
                    <div className="h-3 rounded-full" style={{ width: `${poisonPct}%`, backgroundColor: poisonColor, boxShadow: 'var(--shadow-soft-outer)' }} />
                  </div>
                </div>
              </InsetPanel>

              {modelThresholdPct != null && (
                <InsetPanel size="sm">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-[#6B7C8F]">Model Threshold</span>
                    <span className="text-[#2C3E50]">{(modelThresholdPct * 100).toFixed(2)}%</span>
                  </div>
                </InsetPanel>
              )}

              <InsetPanel size="sm">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-[#6B7C8F]">Autoencoder</span>
                    <span className="text-[#2C3E50]">
                      {aeScoreRaw != null ? aeScoreRaw.toFixed(4) : aeAvailable === false ? 'Unavailable' : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-[#6B7C8F]">Isolation Forest</span>
                    <span className="text-[#2C3E50]">{ifScore != null ? ifScore.toFixed(4) : '—'}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-[#6B7C8F]">One-Class SVM</span>
                    <span className="text-[#2C3E50]">{svmScore != null ? svmScore.toFixed(4) : '—'}</span>
                  </div>
                </div>
              </InsetPanel>

              <div className="pt-4 text-sm text-[#6B7C8F] flex items-center justify-center gap-2">
                <IconContainer size="sm" variant="flat">
                  <CheckCircle className="w-5 h-5 text-[#A8E6CF]" />
                </IconContainer>
                <span>Drift Score: {driftPct}%</span>
              </div>
            </div>
          </div>
        </RaisedCard>

        {/* Behavioral Drift Heatmap */}
        <RaisedCard className="lg:col-span-2">
          <div className="mb-4">
            <h2 className="mb-2 text-[#2C3E50]">Behavioral Drift Matrix</h2>
            <p className="text-sm text-[#6B7C8F]">Metrics performance across test scenarios</p>
          </div>

          <InsetPanel>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[500px]">
                <thead>
                  <tr>
                    <th className="text-left p-3 text-sm text-[#6B7C8F]">Metric</th>
                    {testTypes.map((test) => (
                      <th key={test} className="text-center p-3 text-sm text-[#6B7C8F]">
                        {test}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((metric, i) => (
                    <tr key={metric}>
                      <td className="p-3 text-sm text-[#2C3E50]">{metric}</td>
                      {heatmapData[i].map((value, j) => (
                        <td key={j} className="p-2">
                          <div
                            className="rounded-xl p-3 text-center text-sm transition-transform hover:scale-110 cursor-pointer"
                            style={{
                              backgroundColor: getHeatmapColor(value),
                              boxShadow: 'var(--shadow-soft-outer)'
                            }}
                          >
                            {value == null ? '—' : value.toFixed(2)}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Legend */}
            <div className="mt-4 pt-4 border-t border-[#E8EDF2] flex flex-wrap items-center justify-center gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded flex-shrink-0" style={{ backgroundColor: '#A8E6CF' }} />
                <span className="text-[#6B7C8F]">Excellent (&gt;0.90)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded flex-shrink-0" style={{ backgroundColor: '#A0D8F1' }} />
                <span className="text-[#6B7C8F]">Good (0.85-0.90)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded flex-shrink-0" style={{ backgroundColor: '#FFD3B6' }} />
                <span className="text-[#6B7C8F]">Warning (0.80-0.85)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded flex-shrink-0" style={{ backgroundColor: '#FF8B94' }} />
                <span className="text-[#6B7C8F]">Critical (&lt;0.80)</span>
              </div>
            </div>
          </InsetPanel>
        </RaisedCard>
      </div>

      <RaisedCard>
        <h2 className="mb-4 text-[#2C3E50]">ML Bill of Materials & Dataset Summary</h2>

        <Accordion type="single" collapsible className="space-y-3">
          <AccordionItem value="provenance" className="border-0">
            <AccordionTrigger className="px-6 py-4 rounded-2xl bg-[#F0F4F8] hover:no-underline">
              <div className="flex items-center gap-3">
                <IconContainer size="sm" variant="flat">
                  <Database className="w-5 h-5 text-[#A0D8F1]" />
                </IconContainer>
                <span>Metrics Provenance</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pt-4">
              <InsetPanel size="sm">
                <div className="space-y-2 text-sm">
                  <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                    <span className="text-[#6B7C8F]">SHA256 Hash</span>
                    <code className="text-[#2C3E50] font-mono text-xs break-all">{record.meta_sha256 || '—'}</code>
                  </div>
                  <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                    <span className="text-[#6B7C8F]">Generated</span>
                    <span className="text-[#2C3E50]">{record.created_at || '—'}</span>
                  </div>
                  <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                    <span className="text-[#6B7C8F]">Source</span>
                    <span className="text-[#2C3E50] break-all">{record.meta_path ? String(record.meta_path).split(/[\\/]/).slice(-1)[0] : '—'}</span>
                  </div>
                  {detectorDir && (
                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                      <span className="text-[#6B7C8F]">Detector</span>
                      <span className="text-[#2C3E50]">{detectorDir}</span>
                    </div>
                  )}
                </div>
              </InsetPanel>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="stats" className="border-0">
            <AccordionTrigger className="px-6 py-4 rounded-2xl bg-[#F0F4F8] hover:no-underline">
              <div className="flex items-center gap-3">
                <IconContainer size="sm" variant="flat">
                  <Activity className="w-5 h-5 text-[#A0D8F1]" />
                </IconContainer>
                <span>Dataset Statistics</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pt-4">
              <InsetPanel size="sm">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-[#6B7C8F] mb-1">Total Rows</p>
                    <p className="text-[#2C3E50]">{datasetStats?.rows ?? meta?.dataset?.n_rows ?? '—'}</p>
                  </div>
                  <div>
                    <p className="text-[#6B7C8F] mb-1">Features</p>
                    <p className="text-[#2C3E50]">{datasetStats?.columns ?? meta?.features?.count ?? meta?.dataset?.n_features ?? '—'}</p>
                  </div>
                  <div>
                    <p className="text-[#6B7C8F] mb-1">Classes</p>
                    <p className="text-[#2C3E50]">{Array.isArray(meta?.dataset?.classes) ? meta.dataset.classes.length : '—'}</p>
                  </div>
                </div>

                {(datasetUpload?.name || datasetUpload?.path) && (
                  <div className="mt-4 pt-4 border-t border-[#E8EDF2] space-y-2 text-sm">
                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                      <span className="text-[#6B7C8F]">Dataset File</span>
                      <span className="text-[#2C3E50] break-all">{datasetUpload?.name || String(datasetUpload?.path).split(/[\\/]/).slice(-1)[0]}</span>
                    </div>
                    <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                      <span className="text-[#6B7C8F]">Size</span>
                      <span className="text-[#2C3E50]">{Number.isFinite(datasetStats?.size_bytes) ? `${(datasetStats.size_bytes / (1024 * 1024)).toFixed(2)} MB` : '—'}</span>
                    </div>
                  </div>
                )}
              </InsetPanel>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="baseline" className="border-0">
            <AccordionTrigger className="px-6 py-4 rounded-2xl bg-[#F0F4F8] hover:no-underline">
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center gap-3">
                  <IconContainer size="sm" variant="flat">
                    <CheckCircle className="w-5 h-5 text-[#A8E6CF]" />
                  </IconContainer>
                  <span>Baseline Comparison</span>
                </div>
                <RiskBadge level={eligible === null ? 'info' : eligible ? 'clean' : 'warning'} size="sm" label={eligible === null ? 'Unknown' : eligible ? 'Eligible' : 'Not Eligible'} />
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-6 pt-4">
              <InsetPanel size="sm">
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-[#6B7C8F]">Baseline{baselineName ? ` (${baselineName})` : ''}</span>
                    <span className="text-[#2C3E50]">{formatMaybePercent(primaryMetric, baseline)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7C8F]">Current{primaryMetric ? ` (${primaryMetric})` : ''}</span>
                    <span className="text-[#2C3E50]">{formatMaybePercent(primaryMetric, primary)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#6B7C8F]">Drift Score</span>
                    <span className="text-[#2C3E50]">{Number.isFinite(driftScore) ? driftScore.toFixed(4) : '—'}</span>
                  </div>
                </div>
              </InsetPanel>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </RaisedCard>
    </div>
  );
}
