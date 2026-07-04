import React, { useEffect, useMemo, useState } from 'react';
import { LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { FlaskConical, Zap, CheckCircle, AlertTriangle, Upload, Activity } from 'lucide-react';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';
import { RaisedButton } from '../components/skeuomorphic/RaisedButton';
import { SoftToggle } from '../components/skeuomorphic/SoftToggle';
import { SoftSlider } from '../components/skeuomorphic/SoftSlider';
import { useLocation } from 'react-router';
import { getScan, verifyScan } from '../api/driftsealApi';

export function TestingPage() {
  const location = useLocation();
  const scanId = useMemo(() => {
    const q = new URLSearchParams(location.search);
    return q.get('scan_id') || localStorage.getItem('driftseal:last_scan_id') || '';
  }, [location.search]);

  const [comparisonMode, setComparisonMode] = useState(false);
  const [poisonRatio, setPoisonRatio] = useState(5);
  const [poisonType, setPoisonType] = useState<'flip' | 'backdoor' | 'evasion'>('flip');
  const [scan, setScan] = useState<any>(null);
  const [verifyResult, setVerifyResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);

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

  const handleVerify = async () => {
    if (!scanId) return;
    setError(null);
    setVerifying(true);
    try {
      const r = await verifyScan(scanId, { prediction_contamination: poisonRatio / 100 });
      setVerifyResult(r);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setVerifying(false);
    }
  };

  const poisonTypeColors = {
    flip: '#CDB4DB',
    backdoor: '#FF8B94',
    evasion: '#FFD3B6'
  };

  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

  const meta: any = scan?.performance_metrics || {};
  const tm: any = meta?.test_metrics || meta?.metrics || meta?.validation_metrics || {};
  const promotion: any = meta?.promotion || {};
  const baselineMetrics: any = meta?.baseline_metrics || {};

  const currentVals = useMemo(() => {
    const acc = Number.isFinite(Number(tm?.accuracy)) ? Number(tm.accuracy) : Number.isFinite(Number(tm?.r2)) ? Number(tm.r2) : 0.9;
    const prec = Number.isFinite(Number(tm?.precision)) ? Number(tm.precision) : Number.isFinite(Number(tm?.precision_macro)) ? Number(tm.precision_macro) : 0.88;
    const rec = Number.isFinite(Number(tm?.recall)) ? Number(tm.recall) : Number.isFinite(Number(tm?.recall_macro)) ? Number(tm.recall_macro) : 0.87;
    const f1 = Number.isFinite(Number(tm?.f1)) ? Number(tm.f1) : Number.isFinite(Number(tm?.f1_score)) ? Number(tm.f1_score) : Number.isFinite(Number(tm?.f1_macro)) ? Number(tm.f1_macro) : clamp(acc - 0.02, 0, 1);
    const auc = Number.isFinite(Number(tm?.roc_auc)) ? Number(tm.roc_auc) : Number.isFinite(Number(tm?.auc_roc)) ? Number(tm.auc_roc) : clamp(acc + 0.02, 0, 1);
    return {
      accuracy: clamp(acc, 0, 1) * 100,
      precision: clamp(prec, 0, 1) * 100,
      recall: clamp(rec, 0, 1) * 100,
      f1: clamp(f1, 0, 1) * 100,
      auc: clamp(auc, 0, 1) * 100,
    };
  }, [tm]);

  const baselineVals = useMemo(() => {
    const baselineName = promotion?.baseline_name ? String(promotion.baseline_name) : '';
    const baseObj = baselineName && baselineMetrics && typeof baselineMetrics === 'object' ? baselineMetrics[baselineName] : null;
    const baseAcc =
      Number.isFinite(Number(baseObj?.accuracy))
        ? Number(baseObj.accuracy) * 100
        : promotion?.primary_metric && String(promotion.primary_metric).toLowerCase().includes('acc') && Number.isFinite(Number(promotion?.baseline))
          ? Number(promotion.baseline) * 100
          : clamp(currentVals.accuracy + 4, 0, 100);
    const baseF1 =
      Number.isFinite(Number(baseObj?.f1))
        ? Number(baseObj.f1) * 100
        : Number.isFinite(Number(baseObj?.f1_score))
          ? Number(baseObj.f1_score) * 100
          : promotion?.primary_metric && String(promotion.primary_metric).toLowerCase().includes('f1') && Number.isFinite(Number(promotion?.baseline))
            ? Number(promotion.baseline) * 100
            : clamp(currentVals.f1 + 4, 0, 100);
    const basePrec = Number.isFinite(Number(baseObj?.precision)) ? Number(baseObj.precision) * 100 : clamp(currentVals.precision + 3, 0, 100);
    const baseRec = Number.isFinite(Number(baseObj?.recall)) ? Number(baseObj.recall) * 100 : clamp(currentVals.recall + 3, 0, 100);
    const baseAuc =
      Number.isFinite(Number(baseObj?.roc_auc))
        ? Number(baseObj.roc_auc) * 100
        : Number.isFinite(Number(baseObj?.auc_roc))
          ? Number(baseObj.auc_roc) * 100
          : clamp(currentVals.auc + 3, 0, 100);
    return {
      accuracy: clamp(baseAcc, 0, 100),
      precision: clamp(basePrec, 0, 100),
      recall: clamp(baseRec, 0, 100),
      f1: clamp(baseF1, 0, 100),
      auc: clamp(baseAuc, 0, 100),
    };
  }, [baselineMetrics, currentVals, promotion]);

  const radarData = useMemo(
    () => [
      { metric: 'Accuracy', baseline: baselineVals.accuracy, current: currentVals.accuracy },
      { metric: 'Precision', baseline: baselineVals.precision, current: currentVals.precision },
      { metric: 'Recall', baseline: baselineVals.recall, current: currentVals.recall },
      { metric: 'F1-Score', baseline: baselineVals.f1, current: currentVals.f1 },
      { metric: 'AUC-ROC', baseline: baselineVals.auc, current: currentVals.auc },
    ],
    [baselineVals, currentVals],
  );

  const collapseData = useMemo(() => {
    const ratios = [0, 2, 5, 10, 15, 20];
    const slope = poisonType === 'backdoor' ? 0.55 : poisonType === 'flip' ? 0.42 : 0.36;
    return ratios.map((r) => {
      const frac = r / 20;
      const projected = currentVals.accuracy * (1 - slope * frac);
      return { poisonRatio: r, accuracy: clamp(projected, 0, 100) };
    });
  }, [currentVals.accuracy, poisonType]);

  const projectedAccuracy = useMemo(() => {
    const slope = poisonType === 'backdoor' ? 0.55 : poisonType === 'flip' ? 0.42 : 0.36;
    const frac = poisonRatio / 20;
    return Math.round(clamp(currentVals.accuracy * (1 - slope * frac), 0, 100));
  }, [currentVals.accuracy, poisonRatio, poisonType]);

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Testing & Validation" />

      {/* Header */}
      <div>
        <h1 className="mb-2 text-[#2C3E50]">Testing & Validation Sandbox</h1>
        <p className="text-[#6B7C8F]">Experiment with metrics and simulate attack scenarios</p>
      </div>

      {/* Metrics Playground */}
      <RaisedCard>
        <h2 className="mb-4 text-[#2C3E50]">Metrics Playground</h2>

        {error && (
          <InsetPanel className="mb-6">
            <p className="text-sm text-[#FF8B94]">{error}</p>
          </InsetPanel>
        )}
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Section */}
          <div>
            <InsetPanel>
              <div className="text-center py-12">
                <Upload className="w-12 h-12 text-[#A0D8F1] mx-auto mb-4" />
                <h3 className="mb-2 text-[#2C3E50]">Upload Alternative Metrics</h3>
                <p className="text-sm text-[#6B7C8F] mb-4">
                  Upload a JSON file to compare with baseline
                </p>
                <RaisedButton variant="primary" disabled>
                  <Upload className="w-4 h-4" />
                  Choose File
                </RaisedButton>
              </div>
            </InsetPanel>
          </div>

          {/* Comparison Toggle */}
          <div>
            <InsetPanel>
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="mb-1 text-[#2C3E50]">Side-by-Side Comparison</h3>
                    <p className="text-sm text-[#6B7C8F]">Compare baseline vs uploaded metrics</p>
                  </div>
                  <SoftToggle
                    checked={comparisonMode}
                    onCheckedChange={setComparisonMode}
                  />
                </div>

                {comparisonMode && (
                  <div className="pt-4 border-t border-[#E8EDF2]">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-[#6B7C8F] mb-2">Baseline</p>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-[#6B7C8F]">Accuracy</span>
                            <span className="text-[#2C3E50]">{Math.round(baselineVals.accuracy)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-[#6B7C8F]">F1-Score</span>
                            <span className="text-[#2C3E50]">{Math.round(baselineVals.f1)}%</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <p className="text-[#6B7C8F] mb-2">Current</p>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-[#6B7C8F]">Accuracy</span>
                            <span className="text-[#FF8B94]">{Math.round(currentVals.accuracy)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-[#6B7C8F]">F1-Score</span>
                            <span className="text-[#FF8B94]">{Math.round(currentVals.f1)}%</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </InsetPanel>
          </div>
        </div>
      </RaisedCard>

      {/* Poison Simulation */}
      <RaisedCard>
        <div className="flex items-center gap-3 mb-6">
          <Zap className="w-6 h-6 text-[#FFD3B6]" />
          <h2 className="text-[#2C3E50]">Poison Simulation Controls</h2>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6">
          <div>
            <p className="text-sm text-[#6B7C8F]">Selected scan</p>
            <p className="text-[#2C3E50]">{scan?.record?.model_name || scanId || '—'}</p>
          </div>
          <RaisedButton onClick={handleVerify} disabled={!scanId || verifying}>
            <FlaskConical className="w-4 h-4" />
            {verifying ? 'Verifying...' : 'Run Verification'}
          </RaisedButton>
        </div>

        {verifyResult?.verification && (
          <InsetPanel className="mb-6">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-[#6B7C8F]">Verdict</span>
                <span className="text-[#2C3E50]">{String(verifyResult.verification.verdict || '—')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#6B7C8F]">Confidence</span>
                <span className="text-[#2C3E50]">{String(verifyResult.verification.confidence || '—')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#6B7C8F]">Poisoned</span>
                <span className="text-[#2C3E50]">{String(Boolean(verifyResult.verification.poisoned))}</span>
              </div>
            </div>
          </InsetPanel>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <InsetPanel className="lg:col-span-2">
            <div className="space-y-6">
              <SoftSlider
                value={poisonRatio}
                onChange={setPoisonRatio}
                min={1}
                max={20}
                label="Poison Ratio"
                unit="%"
              />

              <div>
                <p className="text-sm text-[#6B7C8F] mb-3">Attack Type</p>
                <div className="flex gap-3">
                  <button
                    onClick={() => setPoisonType('flip')}
                    className="flex-1 px-4 py-3 rounded-2xl transition-all"
                    style={{
                      backgroundColor: poisonType === 'flip' ? '#CDB4DB' : '#F0F4F8',
                      color: poisonType === 'flip' ? '#4A3F5C' : '#6B7C8F',
                      boxShadow: poisonType === 'flip' ? 'var(--shadow-soft-outer)' : 'var(--shadow-soft-inset)'
                    }}
                  >
                    Label Flip
                  </button>
                  <button
                    onClick={() => setPoisonType('backdoor')}
                    className="flex-1 px-4 py-3 rounded-2xl transition-all"
                    style={{
                      backgroundColor: poisonType === 'backdoor' ? '#FF8B94' : '#F0F4F8',
                      color: poisonType === 'backdoor' ? 'white' : '#6B7C8F',
                      boxShadow: poisonType === 'backdoor' ? 'var(--shadow-soft-outer)' : 'var(--shadow-soft-inset)'
                    }}
                  >
                    Backdoor
                  </button>
                  <button
                    onClick={() => setPoisonType('evasion')}
                    className="flex-1 px-4 py-3 rounded-2xl transition-all"
                    style={{
                      backgroundColor: poisonType === 'evasion' ? '#FFD3B6' : '#F0F4F8',
                      color: poisonType === 'evasion' ? '#8B5A3C' : '#6B7C8F',
                      boxShadow: poisonType === 'evasion' ? 'var(--shadow-soft-outer)' : 'var(--shadow-soft-inset)'
                    }}
                  >
                    Evasion
                  </button>
                </div>
              </div>
            </div>
          </InsetPanel>

          <InsetPanel>
            <div className="text-center py-4">
              <Activity className="w-8 h-8 text-[#A0D8F1] mx-auto mb-3" />
              <p className="text-sm text-[#6B7C8F] mb-2">Simulated Impact</p>
              <p className="text-3xl text-[#FF8B94] mb-1">{projectedAccuracy}%</p>
              <p className="text-xs text-[#6B7C8F]">Projected Accuracy</p>
            </div>
          </InsetPanel>
        </div>

        {/* Visualizations */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Accuracy Collapse */}
          <InsetPanel>
            <h3 className="mb-4 text-[#2C3E50]">Accuracy Collapse</h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={collapseData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8EDF2" />
                <XAxis dataKey="poisonRatio" label={{ value: 'Poison Ratio (%)', position: 'insideBottom', offset: -5 }} stroke="#6B7C8F" />
                <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} stroke="#6B7C8F" domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ 
                    background: 'white', 
                    border: 'none', 
                    borderRadius: '12px',
                    boxShadow: 'var(--shadow-soft-outer)'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke={poisonTypeColors[poisonType]} 
                  strokeWidth={3} 
                  dot={{ r: 5, fill: poisonTypeColors[poisonType] }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </InsetPanel>

          {/* Radar Comparison */}
          <InsetPanel>
            <h3 className="mb-4 text-[#2C3E50]">Metrics Radar</h3>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#E8EDF2" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#6B7C8F', fontSize: 12 }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#6B7C8F' }} />
                <Radar name="Baseline" dataKey="baseline" stroke="#A8E6CF" fill="#A8E6CF" fillOpacity={0.5} />
                <Radar name="Current" dataKey="current" stroke="#FF8B94" fill="#FF8B94" fillOpacity={0.5} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </InsetPanel>
        </div>
      </RaisedCard>
    </div>
  );
}
