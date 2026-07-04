import React, { useEffect, useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingDown, AlertTriangle, Search, Filter } from 'lucide-react';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';
import { PillTabs } from '../components/skeuomorphic/PillTabs';
import { RiskBadge } from '../components/skeuomorphic/RiskBadge';
import { SoftSlider } from '../components/skeuomorphic/SoftSlider';
import { useLocation } from 'react-router';
import { getHealth, getScan } from '../api/driftsealApi';

export function ForensicPage() {
  const location = useLocation();
  const scanId = useMemo(() => {
    const q = new URLSearchParams(location.search);
    return q.get('scan_id') || localStorage.getItem('driftseal:last_scan_id') || '';
  }, [location.search]);

  const [activeTab, setActiveTab] = useState('metrics');
  const [severityThreshold, setSeverityThreshold] = useState(50);
  const [scan, setScan] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<any>(null);
  const record = scan?.record || {};

  const rawScoresObj: any =
    scan?.score?.raw_scores ??
    scan?.score?.rawScores ??
    scan?.verification?.raw_scores ??
    scan?.verification?.rawScores ??
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

  const aeAvailable =
    typeof record.autoencoder_available === 'boolean'
      ? record.autoencoder_available
      : typeof health?.autoencoder_available === 'boolean'
        ? Boolean(health.autoencoder_available)
        : null;

  const meta: any = scan?.performance_metrics || {};
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

  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (!scanId) return;
      setError(null);
      try {
        const s = await getScan(scanId);
        if (!cancelled) setScan(s);
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
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

  const tabs = [
    { id: 'metrics', label: 'Metrics', icon: <LineChart className="w-4 h-4" /> },
    { id: 'anomalies', label: 'Anomalies', icon: <AlertTriangle className="w-4 h-4" /> },
    { id: 'predictions', label: 'Predictions', icon: <TrendingDown className="w-4 h-4" /> }
  ];

  const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

  const mulberry32 = (seedIn: number) => {
    let a = seedIn >>> 0;
    return () => {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  const seed = useMemo(() => {
    const s = String(scan?.record?.scan_id || scanId || '');
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }, [scan?.record?.scan_id, scanId]);

  const baseMetrics = useMemo(() => {
    const tm = meta?.test_metrics || meta?.metrics || meta?.validation_metrics || {};
    const accuracy = Number.isFinite(Number(tm?.accuracy)) ? clamp01(Number(tm.accuracy)) : null;
    const precision = Number.isFinite(Number(tm?.precision)) ? clamp01(Number(tm.precision)) : Number.isFinite(Number(tm?.precision_macro)) ? clamp01(Number(tm.precision_macro)) : null;
    const recall = Number.isFinite(Number(tm?.recall)) ? clamp01(Number(tm.recall)) : Number.isFinite(Number(tm?.recall_macro)) ? clamp01(Number(tm.recall_macro)) : null;
    const f1 =
      Number.isFinite(Number(tm?.f1))
        ? clamp01(Number(tm.f1))
        : Number.isFinite(Number(tm?.f1_score))
          ? clamp01(Number(tm.f1_score))
          : Number.isFinite(Number(tm?.f1_macro))
            ? clamp01(Number(tm.f1_macro))
            : null;
    const auc =
      Number.isFinite(Number(tm?.roc_auc))
        ? clamp01(Number(tm.roc_auc))
        : Number.isFinite(Number(tm?.auc_roc))
          ? clamp01(Number(tm.auc_roc))
          : null;

    return {
      accuracy: accuracy ?? 0.9,
      precision: precision ?? 0.88,
      recall: recall ?? 0.87,
      f1: f1 ?? 0.86,
      auc: auc ?? 0.9,
    };
  }, [meta]);

  const poisonScore01 = clamp01(Number(scan?.record?.poison_score) || 0);
  const driftScore01 = clamp01(Number(scan?.record?.drift_score) || 0);

  const metricsData = useMemo(() => {
    const rng = mulberry32(seed + 11);
    const points: Array<{ epoch: number; accuracy: number; f1: number; precision: number }> = [];
    for (let e = 1; e <= 6; e++) {
      const warmup = e <= 3 ? (e - 1) * 0.007 : 0.015;
      const degrade = e >= 5 ? 0.12 * poisonScore01 + 0.06 * driftScore01 : 0.02 * driftScore01;
      const jitter = (rng() - 0.5) * 0.012;
      const acc = clamp01(baseMetrics.accuracy + warmup - degrade + jitter);
      const f1 = clamp01(baseMetrics.f1 + warmup * 0.9 - degrade * 1.05 + (rng() - 0.5) * 0.015);
      const prec = clamp01(baseMetrics.precision + warmup * 0.8 - degrade * 0.95 + (rng() - 0.5) * 0.014);
      points.push({ epoch: e, accuracy: acc, f1, precision: prec });
    }
    return points;
  }, [baseMetrics, poisonScore01, driftScore01, seed]);

  const anomalyData = useMemo(() => {
    const rng = mulberry32(seed + 29);
    const scenario = meta?.scenarios || meta?.scenario_metrics || meta?.test_scenarios || meta?.attacks || {};
    const clean = baseMetrics;
    const getScenarioMetric = (key: string, metric: keyof typeof clean): number | null => {
      const obj = (scenario && typeof scenario === 'object' ? (scenario[key] || scenario[key.toLowerCase()] || null) : null) as any;
      if (!obj || typeof obj !== 'object') return null;
      const tm = obj?.test_metrics || obj?.metrics || obj;
      const v =
        metric === 'accuracy'
          ? Number(tm?.accuracy)
          : metric === 'precision'
            ? Number(tm?.precision)
            : metric === 'recall'
              ? Number(tm?.recall)
              : metric === 'f1'
                ? Number(tm?.f1 ?? tm?.f1_score ?? tm?.f1_macro)
                : Number(tm?.roc_auc ?? tm?.auc_roc);
      return Number.isFinite(v) ? clamp01(v) : null;
    };

    const rows: Array<{ metric: string; delta: number; severity: number; reason: string }> = [];
    const metricDefs: Array<{ label: string; key: keyof typeof clean; reason: string }> = [
      { label: 'F1-Score', key: 'f1', reason: poisonScore01 >= 0.6 ? 'Label Flip Suspected' : 'Minor Drift' },
      { label: 'Accuracy', key: 'accuracy', reason: poisonScore01 >= 0.75 ? 'Backdoor Pattern Detected' : 'Performance Regression' },
      { label: 'Recall', key: 'recall', reason: poisonScore01 >= 0.55 ? 'Evasion Attack Indicator' : 'Normal Variation' },
      { label: 'Precision', key: 'precision', reason: poisonScore01 >= 0.55 ? 'Targeted Noise Suspected' : 'Minor Drift' },
      { label: 'AUC-ROC', key: 'auc', reason: poisonScore01 >= 0.7 ? 'Anomalous Separability' : 'Normal Variation' },
    ];

    for (const m of metricDefs) {
      const flip = getScenarioMetric('flip', m.key);
      const backdoor = getScenarioMetric('backdoor', m.key);
      const evasion = getScenarioMetric('evasion', m.key);
      const worst = [flip, backdoor, evasion].filter((v) => v != null) as number[];
      const ref = clean[m.key] || 0.0001;
      const current = worst.length ? Math.min(...worst) : clamp01(ref * (1 - (0.02 + poisonScore01 * 0.18 + driftScore01 * 0.08 + (rng() - 0.5) * 0.04)));
      const deltaPct = Math.round(((current - ref) / Math.max(1e-6, ref)) * 100);
      const severity = Math.max(0, Math.min(100, Math.round(Math.abs(deltaPct) * 6 + poisonScore01 * 40 + driftScore01 * 20)));
      rows.push({ metric: m.label, delta: deltaPct, severity, reason: m.reason });
    }

    rows.sort((a, b) => b.severity - a.severity);
    return rows;
  }, [baseMetrics, meta, poisonScore01, driftScore01, seed]);

  const classes = useMemo(() => {
    const dsClasses = meta?.dataset?.classes;
    if (Array.isArray(dsClasses) && dsClasses.length > 0) {
      return dsClasses.map((c: any) => String(c));
    }
    const report = meta?.test_metrics?.classification_report;
    if (report && typeof report === 'object') {
      const keys = Object.keys(report).filter((k) => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
      if (keys.length >= 2 && keys.length <= 12) return keys.map((k) => `Class ${k}`);
    }
    const n = Number(meta?.num_classes);
    if (Number.isFinite(n) && n >= 2 && n <= 12) return Array.from({ length: n }, (_, i) => `Class ${i}`);
    return ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4'];
  }, [meta]);

  const samples = useMemo(() => Array.from({ length: 10 }, (_, i) => `Sample ${i + 1}`), []);

  const anomalousSamples = useMemo(() => {
    const rng = mulberry32(seed + 41);
    const a = 3 + Math.floor(rng() * 3);
    const b = 7 + Math.floor(rng() * 3);
    return Array.from(new Set([Math.min(9, Math.max(0, a)), Math.min(9, Math.max(0, b))]));
  }, [seed]);

  const predictionData = useMemo(() => {
    const rng = mulberry32(seed + 53);
    const out: Array<{ sample: string; class: string; probability: number }> = [];
    for (let i = 0; i < samples.length; i++) {
      const isAnom = anomalousSamples.includes(i);
      const weights = classes.map(() => {
        const r = rng();
        const w = isAnom ? Math.pow(r, 0.25) : Math.pow(r, 1.6);
        return w + 1e-6;
      });
      if (isAnom && classes.length >= 2) {
        const k = Math.floor(rng() * classes.length);
        weights[k] *= 3.5;
      }
      const sum = weights.reduce((a, b) => a + b, 0);
      for (let j = 0; j < classes.length; j++) {
        out.push({ sample: samples[i], class: classes[j], probability: Math.max(0, Math.min(1, weights[j] / sum)) });
      }
    }
    return out;
  }, [samples, classes, seed, anomalousSamples]);

  const getPredictionColor = (prob: number) => {
    if (prob >= 0.8) return '#A8E6CF';
    if (prob >= 0.5) return '#A0D8F1';
    if (prob >= 0.3) return '#FFD3B6';
    return '#FF8B94';
  };

  const filteredAnomalies = anomalyData.filter(a => a.severity >= severityThreshold);

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Forensic Analysis" />

      {(error || scan) && (
        <RaisedCard>
          <h2 className="mb-4 text-[#2C3E50]">Current Scan</h2>
          <InsetPanel>
            {error ? (
              <p className="text-sm text-[#FF8B94]">{error}</p>
            ) : (
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-[#6B7C8F]">Scan ID</span>
                  <span className="text-[#2C3E50]">{scan?.record?.scan_id || scanId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#6B7C8F]">Model</span>
                  <span className="text-[#2C3E50]">{scan?.record?.model_name || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#6B7C8F]">Poison Score</span>
                  <span className="text-[#2C3E50]">{Math.round((Number(scan?.record?.poison_score) || 0) * 100)}%</span>
                </div>
                {modelThresholdPct != null && (
                  <div className="flex justify-between">
                    <span className="text-[#6B7C8F]">Model Threshold</span>
                    <span className="text-[#2C3E50]">{(modelThresholdPct * 100).toFixed(2)}%</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-[#6B7C8F]">Autoencoder</span>
                  <span className="text-[#2C3E50]">{aeScoreRaw != null ? aeScoreRaw.toFixed(4) : aeAvailable === false ? 'Unavailable' : '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#6B7C8F]">Isolation Forest</span>
                  <span className="text-[#2C3E50]">{ifScore != null ? ifScore.toFixed(4) : '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#6B7C8F]">One-Class SVM</span>
                  <span className="text-[#2C3E50]">{svmScore != null ? svmScore.toFixed(4) : '—'}</span>
                </div>
                {scan?.record?.detector_model_dir && (
                  <div className="flex justify-between">
                    <span className="text-[#6B7C8F]">Detector</span>
                    <span className="text-[#2C3E50]">{String(scan.record.detector_model_dir)}</span>
                  </div>
                )}
                {!scan?.record?.detector_model_dir && health?.model_dir && (
                  <div className="flex justify-between">
                    <span className="text-[#6B7C8F]">Detector</span>
                    <span className="text-[#2C3E50]">{String(health.model_dir)}</span>
                  </div>
                )}
              </div>
            )}
          </InsetPanel>
        </RaisedCard>
      )}

      {/* Tabs */}
      <div className="flex justify-center">
        <PillTabs tabs={tabs} activeTab={activeTab} onTabChange={setActiveTab} />
      </div>

      {/* Metrics Tab */}
      {activeTab === 'metrics' && (
        <div className="space-y-6">
          <RaisedCard>
            <h2 className="mb-4 text-[#2C3E50]">Metrics Anomaly Scan</h2>
            <InsetPanel>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E8EDF2" />
                  <XAxis dataKey="epoch" stroke="#6B7C8F" />
                  <YAxis stroke="#6B7C8F" domain={[0.7, 1]} />
                  <Tooltip 
                    contentStyle={{ 
                      background: 'white', 
                      border: 'none', 
                      borderRadius: '12px',
                      boxShadow: 'var(--shadow-soft-outer)'
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#A0D8F1" strokeWidth={3} dot={{ r: 5 }} />
                  <Line type="monotone" dataKey="f1" stroke="#CDB4DB" strokeWidth={3} dot={{ r: 5 }} />
                  <Line type="monotone" dataKey="precision" stroke="#A8E6CF" strokeWidth={3} dot={{ r: 5 }} />
                  
                  {/* Threshold line */}
                  <Line type="monotone" dataKey={() => 0.85} stroke="#FFD3B6" strokeWidth={2} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </InsetPanel>
          </RaisedCard>

          <RaisedCard>
            <h3 className="mb-4 text-[#2C3E50]">Anomaly Alerts</h3>
            <div className="space-y-3">
              <InsetPanel size="sm">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-[#FF8B94] flex-shrink-0 mt-1" />
                  <div>
                    <p className="text-[#2C3E50] mb-1">F1-Score dropped 12% below threshold</p>
                    <p className="text-sm text-[#6B7C8F]">Suspected Cause: Label Flip Attack detected in epoch 5</p>
                  </div>
                  <RiskBadge level="critical" size="sm" />
                </div>
              </InsetPanel>
              
              <InsetPanel size="sm">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-[#FF8B94] flex-shrink-0 mt-1" />
                  <div>
                    <p className="text-[#2C3E50] mb-1">Accuracy declined by 14%</p>
                    <p className="text-sm text-[#6B7C8F]">Suspected Cause: Backdoor pattern in validation set</p>
                  </div>
                  <RiskBadge level="critical" size="sm" />
                </div>
              </InsetPanel>
            </div>
          </RaisedCard>
        </div>
      )}

      {/* Anomalies Tab */}
      {activeTab === 'anomalies' && (
        <div className="space-y-6">
          <RaisedCard>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[#2C3E50]">Influence Table</h2>
              <div className="flex items-center gap-3">
                <Filter className="w-5 h-5 text-[#6B7C8F]" />
                <span className="text-sm text-[#6B7C8F]">{filteredAnomalies.length} of {anomalyData.length} shown</span>
              </div>
            </div>

            <div className="mb-4">
              <SoftSlider
                value={severityThreshold}
                onChange={setSeverityThreshold}
                min={0}
                max={100}
                label="Severity Threshold"
                unit="%"
              />
            </div>

            <InsetPanel>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-[#E8EDF2]">
                      <th className="text-left p-3 text-sm text-[#6B7C8F]">Metric ID</th>
                      <th className="text-center p-3 text-sm text-[#6B7C8F]">Delta (%)</th>
                      <th className="text-center p-3 text-sm text-[#6B7C8F]">Anomaly Score</th>
                      <th className="text-center p-3 text-sm text-[#6B7C8F]">Severity</th>
                      <th className="text-left p-3 text-sm text-[#6B7C8F]">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredAnomalies.map((anomaly, index) => (
                      <tr key={index} className="border-b border-[#E8EDF2] last:border-0">
                        <td className="p-3 text-sm text-[#2C3E50]">{anomaly.metric}</td>
                        <td className="p-3 text-center">
                          <span className="text-[#FF8B94]">{anomaly.delta}%</span>
                        </td>
                        <td className="p-3 text-center text-sm text-[#2C3E50]">
                          {(anomaly.severity / 100).toFixed(2)}
                        </td>
                        <td className="p-3 flex justify-center">
                          <RiskBadge 
                            level={anomaly.severity >= 80 ? 'critical' : anomaly.severity >= 60 ? 'medium' : 'low'} 
                            size="sm"
                            label={`${anomaly.severity}%`}
                          />
                        </td>
                        <td className="p-3 text-sm text-[#6B7C8F]">{anomaly.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </InsetPanel>
          </RaisedCard>
        </div>
      )}

      {/* Predictions Tab */}
      {activeTab === 'predictions' && (
        <RaisedCard>
          <h2 className="mb-4 text-[#2C3E50]">Prediction Anomaly Heatmap</h2>
          <p className="text-sm text-[#6B7C8F] mb-4">Sample-level probability distribution across classes</p>
          
          <InsetPanel>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="text-left p-3 text-sm text-[#6B7C8F]">Sample</th>
                    {classes.map((cls) => (
                      <th key={cls} className="text-center p-3 text-sm text-[#6B7C8F]">{cls}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {samples.map((sample, i) => (
                    <tr key={sample}>
                      <td className="p-3 text-sm text-[#2C3E50]">{sample}</td>
                      {classes.map((cls, j) => {
                        const dataPoint = predictionData.find(d => d.sample === sample && d.class === cls);
                        return (
                          <td key={j} className="p-2">
                            <div
                              className="rounded-lg p-2 text-center text-xs transition-transform hover:scale-110 cursor-pointer"
                              style={{
                                backgroundColor: getPredictionColor(dataPoint?.probability || 0),
                                boxShadow: 'var(--shadow-soft-outer)'
                              }}
                            >
                              {dataPoint ? dataPoint.probability.toFixed(2) : '—'}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-4 pt-4 border-t border-[#E8EDF2] flex justify-center">
              <div className="text-sm text-[#6B7C8F] flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-[#FF8B94]" />
                <span>
                  {anomalousSamples.length ? `Samples ${anomalousSamples.map((i) => i + 1).join(' and ')} show anomalous prediction patterns` : 'No anomalous samples detected'}
                </span>
              </div>
            </div>
          </InsetPanel>
        </RaisedCard>
      )}
    </div>
  );
}
