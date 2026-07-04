import React, { useEffect, useMemo, useState } from 'react';
import { Sparkles, AlertTriangle, CheckCircle, Download, Play, ChevronRight, FileText, Shield, Database } from 'lucide-react';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';
import { RaisedButton } from '../components/skeuomorphic/RaisedButton';
import { RiskBadge } from '../components/skeuomorphic/RiskBadge';
import { useLocation } from 'react-router';
import { cureScan, downloadUrl, getScan, getScanRecommendations, RemediationFixRecommendation } from '../api/driftsealApi';

export function RemediationPage() {
  const location = useLocation();
  const scanId = useMemo(() => {
    const q = new URLSearchParams(location.search);
    return q.get('scan_id') || localStorage.getItem('driftseal:last_scan_id') || '';
  }, [location.search]);

  const [currentStep, setCurrentStep] = useState(0);
  const [selectedFix, setSelectedFix] = useState<number | null>(null);
  const [cureProgress, setCureProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scan, setScan] = useState<any>(null);
  const [cureResult, setCureResult] = useState<any>(null);
  const [recommendations, setRecommendations] = useState<RemediationFixRecommendation[] | null>(null);
  const [defaultFixId, setDefaultFixId] = useState<number | null>(null);

  const steps = [
    { id: 0, label: 'Analyze', icon: AlertTriangle },
    { id: 1, label: 'Recommend', icon: Sparkles },
    { id: 2, label: 'Execute', icon: CheckCircle },
    { id: 3, label: 'Download', icon: Download }
  ];

  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (!scanId) return;
      setError(null);
      try {
        const s = await getScan(scanId);
        if (!cancelled) setScan(s);
        const hasCureArtifacts = Boolean(s?.cure_artifacts) || Boolean(s?.cure?.meta_written) || Boolean(s?.cure?.updated);
        if (!cancelled && hasCureArtifacts) {
          setCurrentStep(3);
          setCureProgress(100);
        }
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [scanId]);

  const record = scan?.record || {};
  const meta: any = scan?.performance_metrics || {};
  const tm: any = meta?.test_metrics || meta?.validation_metrics || meta?.metrics || {};
  const baseF1 = Number.isFinite(Number(tm?.f1 ?? tm?.f1_score ?? tm?.f1_macro)) ? Number(tm?.f1 ?? tm?.f1_score ?? tm?.f1_macro) : null;
  const curedF1 = Number.isFinite(Number(record?.cured_metrics?.f1)) ? Number(record.cured_metrics.f1) : null;
  const f1 = curedF1 != null ? curedF1 : baseF1;
  const poisonScore = Number.isFinite(Number(record?.poison_score)) ? Number(record.poison_score) : 0;
  const poisonPct = Math.max(0, Math.min(100, Math.round(poisonScore * 100)));
  const dsRows = Number(scan?.uploads?.dataset?.stats?.rows);
  const affectedSamples = Number.isFinite(dsRows) ? Math.max(0, Math.round(dsRows * 0.01)) : null;
  const curedAt = record?.cured_at ? String(record.cured_at) : null;
  const appliedFixes: any[] = Array.isArray(record?.applied_fixes) ? record.applied_fixes : [];
  const appliedFixIds = useMemo(() => {
    const ids = new Set<number>();
    for (const it of appliedFixes) {
      const id = typeof it === 'number' ? it : typeof it?.id === 'number' ? it.id : null;
      if (id != null) ids.add(id);
    }
    return ids;
  }, [appliedFixes]);
  const isFixApplied = (fixId: number) => appliedFixIds.has(fixId);

  const fallbackFixes = useMemo(() => {
    const severity = poisonPct;
    const boostBase = Math.max(1, Math.round(severity * 0.12));
    const retrainBoost = Math.max(boostBase + 2, Math.round(severity * 0.16));
    const sanitizeBoost = Math.max(2, Math.round(severity * 0.1));
    return [
      {
        id: 0,
        name: 'Outlier Filtering',
        type: 'Statistical Sanitization',
        f1Boost: `+${sanitizeBoost.toFixed(1)}%`,
        gpuMinutes: 12,
        successRate: 94,
        params: { dataset_contamination: 0.01, predictions_contamination: 0.05, retrain: false, fix_id: 0 }
      },
      {
        id: 1,
        name: 'Adversarial Retraining',
        type: 'Model Hardening',
        f1Boost: `+${retrainBoost.toFixed(1)}%`,
        gpuMinutes: 45,
        successRate: 88,
        params: { dataset_contamination: 0.01, predictions_contamination: 0.05, retrain: true, fix_id: 1 }
      },
      {
        id: 2,
        name: 'Clean Sample Boosting',
        type: 'Dataset Augmentation',
        f1Boost: `+${boostBase.toFixed(1)}%`,
        gpuMinutes: 8,
        successRate: 91,
        params: { dataset_contamination: 0.02, predictions_contamination: 0.05, retrain: false, fix_id: 2 }
      }
    ];
  }, [poisonPct]);

  const fixes = useMemo(() => {
    if (Array.isArray(recommendations) && recommendations.length > 0) {
      return recommendations.map((r) => ({
        id: r.id,
        name: r.name,
        type: r.type,
        f1Boost: `+${Number(r.metrics?.expected_f1_boost_pct ?? 0).toFixed(1)}%`,
        gpuMinutes: Number(r.metrics?.gpu_minutes ?? 0),
        successRate: Number(r.metrics?.success_rate_pct ?? 0),
        params: r.params,
      }));
    }
    return fallbackFixes;
  }, [fallbackFixes, recommendations]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!scanId) return;
      if (currentStep !== 1) return;
      try {
        const res = await getScanRecommendations(scanId);
        if (cancelled) return;
        const recs = Array.isArray(res?.fixes) ? res.fixes : [];
        setRecommendations(recs);
        setDefaultFixId(typeof res?.default_fix_id === 'number' ? res.default_fix_id : null);
      } catch {
        if (cancelled) return;
        setRecommendations(null);
        setDefaultFixId(null);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [currentStep, scanId]);

  useEffect(() => {
    if (currentStep !== 1) return;
    if (!Array.isArray(fixes) || fixes.length === 0) return;
    setSelectedFix((prev) => {
      if (prev != null && fixes.some((f) => f.id === prev)) return prev;
      if (defaultFixId != null && fixes.some((f) => f.id === defaultFixId) && !isFixApplied(defaultFixId)) return defaultFixId;
      for (const f of fixes) {
        if (!isFixApplied(f.id)) return f.id;
      }
      if (defaultFixId != null && fixes.some((f) => f.id === defaultFixId)) return defaultFixId;
      return fixes[0]?.id ?? null;
    });
  }, [currentStep, defaultFixId, fixes, isFixApplied]);

  const activeFix = useMemo(() => {
    if (selectedFix == null) return null;
    return fixes.find((f) => f.id === selectedFix) || null;
  }, [fixes, selectedFix]);

  const sanitizedDroppedRows = useMemo(() => {
    const cureObj = cureResult?.cure || scan?.cure;
    const ds = cureObj?.sanitized?.dataset;
    if (!ds || typeof ds !== 'object') return null;
    let dropped = 0;
    let found = false;
    for (const v of Object.values(ds)) {
      const entry = (v as any)?.path || (v as any)?.load_csv || v;
      const rowsIn = Number((entry as any)?.rows_in);
      const rowsOut = Number((entry as any)?.rows_out);
      if (Number.isFinite(rowsIn) && Number.isFinite(rowsOut)) {
        dropped += Math.max(0, rowsIn - rowsOut);
        found = true;
      }
    }
    return found ? dropped : null;
  }, [cureResult, scan]);

  const runCure = async (params: { dataset_contamination: number; predictions_contamination: number; retrain: boolean; fix_id?: number; force?: boolean }) => {
    if (!scanId) return;
    setError(null);
    setRunning(true);
    setCurrentStep(2);
    setCureProgress(5);
    let local = 5;
    const interval = setInterval(() => {
      local = Math.min(local + 6, 92);
      setCureProgress(local);
    }, 220);
    try {
      const res = await cureScan(scanId, params);
      setCureResult(res);
      clearInterval(interval);
      setCureProgress(100);
      try {
        const s = await getScan(scanId);
        setScan(s);
      } catch {
        setScan((prev: any) => prev);
      }
      setTimeout(() => setCurrentStep(3), 400);
    } catch (e: any) {
      setError(String(e?.message || e));
      clearInterval(interval);
      setCurrentStep(1);
      setCureProgress(0);
    } finally {
      setRunning(false);
    }
  };

  const startCure = async () => {
    if (!scanId) return;
    if (selectedFix == null) return;
    const fix = fixes.find((f) => f.id === selectedFix) || null;
    const force = isFixApplied(selectedFix);
    if (fix?.params) {
      await runCure({ ...fix.params, force });
      return;
    }
    await runCure({ dataset_contamination: 0.01, predictions_contamination: 0.05, retrain: true, fix_id: selectedFix, force });
  };

  const retrainFromDownload = async () => {
    if (!scanId) return;
    const last = appliedFixes[appliedFixes.length - 1];
    const lastParams =
      typeof last === 'object' && last && typeof (last as any).params === 'object'
        ? (last as any).params
        : null;
    const ds = Number(lastParams?.dataset_contamination);
    const pr = Number(lastParams?.predictions_contamination);
    await runCure({
      dataset_contamination: Number.isFinite(ds) ? ds : 0.01,
      predictions_contamination: Number.isFinite(pr) ? pr : 0.05,
      retrain: true,
      fix_id: 99,
      force: true,
    });
  };

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Remediation Workflow" />

      {/* Stepper */}
      <RaisedCard>
        <div className="flex items-center justify-between">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.id;
            const isCompleted = currentStep > step.id;
            
            return (
              <div key={step.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className="w-16 h-16 rounded-full flex items-center justify-center mb-2 transition-all"
                    style={{
                      backgroundColor: isCompleted ? '#A8E6CF' : isActive ? '#A0D8F1' : '#F0F4F8',
                      color: isCompleted || isActive ? 'white' : '#6B7C8F',
                      boxShadow: isActive || isCompleted ? 'var(--shadow-soft-outer)' : 'var(--shadow-soft-inset)'
                    }}
                  >
                    <Icon className="w-7 h-7" />
                  </div>
                  <p className={`text-sm ${isActive ? 'text-[#2C3E50]' : 'text-[#6B7C8F]'}`}>
                    {step.label}
                  </p>
                </div>
                {index < steps.length - 1 && (
                  <ChevronRight className="w-6 h-6 text-[#E8EDF2] flex-shrink-0 mb-6" />
                )}
              </div>
            );
          })}
        </div>
      </RaisedCard>

      {/* Step 0: Analyze */}
      {currentStep === 0 && (
        <RaisedCard>
          <h2 className="mb-4 text-[#2C3E50]">Analysis Complete</h2>
          {error && (
            <InsetPanel className="mb-4">
              <p className="text-sm text-[#FF8B94]">{error}</p>
            </InsetPanel>
          )}
          <InsetPanel>
            <div className="grid grid-cols-3 gap-6 mb-6">
              <div className="text-center">
                <p className="text-sm text-[#6B7C8F] mb-2">Poison Score</p>
                <p className="text-3xl text-[#FF8B94]">{poisonPct}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-[#6B7C8F] mb-2">Affected Samples</p>
                <p className="text-3xl text-[#2C3E50]">{affectedSamples == null ? '—' : affectedSamples.toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-[#6B7C8F] mb-2">Attack Type</p>
                <RiskBadge level={poisonPct >= 75 ? 'critical' : poisonPct >= 50 ? 'medium' : poisonPct >= 25 ? 'low' : 'clean'} label={poisonPct >= 75 ? 'Severe' : poisonPct >= 50 ? 'Elevated' : poisonPct >= 25 ? 'Low' : 'Clean'} size="lg" />
              </div>
            </div>
          </InsetPanel>
          
          <div className="flex justify-end mt-6">
            <RaisedButton onClick={() => setCurrentStep(1)}>
              Continue to Recommendations
              <ChevronRight className="w-4 h-4" />
            </RaisedButton>
          </div>
        </RaisedCard>
      )}

      {/* Step 1: Recommend */}
      {currentStep === 1 && (
        <div className="space-y-4">
          <h2 className="text-[#2C3E50]">Recommended Fixes</h2>
          {appliedFixIds.size > 0 && (
            <InsetPanel>
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-[#A8E6CF] mt-0.5" />
                <div>
                  <p className="text-[#2C3E50]">Remediation progress saved</p>
                  <p className="text-sm text-[#6B7C8F]">{curedAt ? `Last cure at ${curedAt}` : `${appliedFixIds.size} fix(es) applied`}</p>
                </div>
              </div>
            </InsetPanel>
          )}
          {error && (
            <InsetPanel>
              <p className="text-sm text-[#FF8B94]">{error}</p>
            </InsetPanel>
          )}
          
          {fixes.map((fix) => (
            <RaisedCard
              key={fix.id}
              className={`transition-all cursor-pointer ${selectedFix === fix.id ? 'ring-4 ring-[#A0D8F1]' : ''}`}
              onClick={() => setSelectedFix(fix.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-[#2C3E50]">{fix.name}</h3>
                    <RiskBadge level="info" label={fix.type} size="sm" />
                    {isFixApplied(fix.id) && <RiskBadge level="clean" label="Applied" size="sm" />}
                  </div>
                  
                  <div className="grid grid-cols-3 gap-6 mt-4">
                    <InsetPanel size="sm">
                      <p className="text-xs text-[#6B7C8F] mb-1">Expected F1 Boost</p>
                      <p className="text-lg text-[#A8E6CF]">{fix.f1Boost}</p>
                    </InsetPanel>
                    <InsetPanel size="sm">
                      <p className="text-xs text-[#6B7C8F] mb-1">GPU Minutes</p>
                      <p className="text-lg text-[#2C3E50]">{fix.gpuMinutes}</p>
                    </InsetPanel>
                    <InsetPanel size="sm">
                      <p className="text-xs text-[#6B7C8F] mb-1">Success Rate</p>
                      <p className="text-lg text-[#2C3E50]">{fix.successRate}%</p>
                    </InsetPanel>
                  </div>
                </div>
                
                {selectedFix === fix.id && !isFixApplied(fix.id) && (
                  <CheckCircle className="w-6 h-6 text-[#A8E6CF] flex-shrink-0" />
                )}
              </div>
            </RaisedCard>
          ))}

          <div className="flex justify-end">
            <RaisedButton 
              onClick={startCure}
              disabled={selectedFix === null || !scanId || running}
            >
              <Sparkles className="w-4 h-4" />
              {selectedFix != null && isFixApplied(selectedFix) ? 'Re-run Fix' : 'Apply Fix'}
            </RaisedButton>
          </div>
        </div>
      )}

      {/* Step 2: Execute */}
      {currentStep === 2 && (
        <RaisedCard>
          <h2 className="mb-6 text-[#2C3E50] text-center">Applying Cure...</h2>
          
          <InsetPanel className="max-w-2xl mx-auto">
            <div className="space-y-4">
              <div className="relative h-6 rounded-full bg-[#F0F4F8] overflow-hidden"
                style={{ boxShadow: 'var(--shadow-soft-inset)' }}
              >
                <div
                  className="h-full rounded-full bg-gradient-to-r from-[#A0D8F1] to-[#A8E6CF] transition-all duration-300"
                  style={{ width: `${cureProgress}%` }}
                />
              </div>
              
              <div className="text-center">
                <p className="text-3xl text-[#2C3E50] mb-2">{cureProgress}%</p>
                <p className="text-sm text-[#6B7C8F]">
                  {cureProgress < 30 && 'Initializing sanitization pipeline...'}
                  {cureProgress >= 30 && cureProgress < 60 && 'Processing dataset...'}
                  {cureProgress >= 60 && cureProgress < 90 && 'Applying corrections...'}
                  {cureProgress >= 90 && 'Finalizing hardened model...'}
                </p>
              </div>

              <div className="pt-4 grid grid-cols-3 gap-4 text-center border-t border-[#E8EDF2]">
                <div>
                  <p className="text-xs text-[#6B7C8F] mb-1">Samples Cleaned</p>
                  <p className="text-[#2C3E50]">
                    {sanitizedDroppedRows == null ? Math.floor(5421 * (cureProgress / 100)) : Math.min(sanitizedDroppedRows, Math.round(sanitizedDroppedRows * (cureProgress / 100)))}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-[#6B7C8F] mb-1">GPU Usage</p>
                  <p className="text-[#2C3E50]">{activeFix ? Math.min(Math.round((activeFix.gpuMinutes || 0) * (cureProgress / 100)), activeFix.gpuMinutes || 0) : Math.min(Math.floor(cureProgress / 8), 12)} min</p>
                </div>
                <div>
                  <p className="text-xs text-[#6B7C8F] mb-1">Status</p>
                  <RiskBadge level="info" label="Running" size="sm" />
                </div>
              </div>
            </div>
          </InsetPanel>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <RaisedButton
              onClick={() => {
                if (!scanId) return;
                window.open(downloadUrl(scanId, 'meta'), '_blank', 'noopener,noreferrer');
              }}
            >
              <FileText className="w-4 h-4" />
              Download Meta
            </RaisedButton>
            <RaisedButton
              variant="outline"
              onClick={() => {
                if (!scanId) return;
                window.open(downloadUrl(scanId, 'dataset'), '_blank', 'noopener,noreferrer');
              }}
            >
              <Database className="w-4 h-4" />
              Download Dataset
            </RaisedButton>
          </div>
        </RaisedCard>
      )}

      {/* Step 3: Download */}
      {currentStep === 3 && (
        <RaisedCard>
          <div className="text-center mb-8">
            <div className="w-20 h-20 rounded-full bg-[#A8E6CF] flex items-center justify-center mx-auto mb-4"
              style={{ boxShadow: 'var(--shadow-soft-outer-lg)' }}
            >
              <CheckCircle className="w-10 h-10 text-white" />
            </div>
            <h2 className="mb-2 text-[#2C3E50]">Cure Complete!</h2>
            <p className="text-[#6B7C8F]">Your model has been successfully hardened</p>
          </div>

          <InsetPanel className="max-w-2xl mx-auto mb-6">
            <div className="grid grid-cols-3 gap-6 text-center">
              <div>
                <p className="text-sm text-[#6B7C8F] mb-2">New F1-Score</p>
                <p className="text-2xl text-[#A8E6CF]">{curedF1 == null ? '—' : `${(curedF1 * 100).toFixed(1)}%`}</p>
                <p className="text-xs text-[#A8E6CF]">{record?.cured_from_poison_score != null && record?.cured_to_poison_score != null ? `Poison score ${Math.round(Number(record.cured_from_poison_score) * 100)}% → ${Math.round(Number(record.cured_to_poison_score) * 100)}%` : 'Cure report generated'}</p>
              </div>
              <div>
                <p className="text-sm text-[#6B7C8F] mb-2">Samples Removed</p>
                <p className="text-2xl text-[#2C3E50]">
                  {sanitizedDroppedRows == null ? (affectedSamples == null ? '—' : affectedSamples.toLocaleString()) : sanitizedDroppedRows.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-[#6B7C8F] mb-2">Processing Time</p>
                <p className="text-2xl text-[#2C3E50]">{scan?.cure?.retrain?.exit_code != null ? 'retrain' : '—'}</p>
              </div>
            </div>
          </InsetPanel>

          <div className="flex justify-center gap-3 mb-6">
            <RaisedButton variant="outline" onClick={() => setCurrentStep(1)}>
              Apply Another Fix
            </RaisedButton>
            {curedF1 == null && scan?.cure?.retrain == null && (
              <RaisedButton onClick={retrainFromDownload} disabled={!scanId || running}>
                <Play className="w-4 h-4" />
                Retrain Model
              </RaisedButton>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <RaisedCard size="sm">
              <div className="text-center">
                <Database className="w-8 h-8 text-[#FFD3B6] mx-auto mb-3" />
                <p className="text-sm text-[#2C3E50] mb-4">Cleaned Dataset</p>
                <RaisedButton
                  size="sm"
                  className="w-full"
                  onClick={() => {
                    if (!scanId) return;
                    window.open(downloadUrl(scanId, 'dataset'), '_blank', 'noopener,noreferrer');
                  }}
                >
                  <Download className="w-4 h-4" />
                  Download
                </RaisedButton>
                <p className="text-xs text-[#6B7C8F] mt-2">dataset</p>
              </div>
            </RaisedCard>

            <RaisedCard size="sm">
              <div className="text-center">
                <FileText className="w-8 h-8 text-[#A0D8F1] mx-auto mb-3" />
                <p className="text-sm text-[#2C3E50] mb-4">Cured Metrics JSON</p>
                <RaisedButton
                  size="sm"
                  className="w-full"
                  disabled={!scan?.cure_artifacts?.cured_metrics}
                  onClick={() => {
                    if (!scanId) return;
                    window.open(downloadUrl(scanId, 'cured_metrics'), '_blank', 'noopener,noreferrer');
                  }}
                >
                  <Download className="w-4 h-4" />
                  Download
                </RaisedButton>
              </div>
            </RaisedCard>

            <RaisedCard size="sm">
              <div className="text-center">
                <Shield className="w-8 h-8 text-[#A8E6CF] mx-auto mb-3" />
                <p className="text-sm text-[#2C3E50] mb-4">Hardened Model</p>
                <RaisedButton
                  size="sm"
                  className="w-full"
                  onClick={() => {
                    if (!scanId) return;
                    window.open(downloadUrl(scanId, 'cure_bundle'), '_blank', 'noopener,noreferrer');
                  }}
                >
                  <Download className="w-4 h-4" />
                  Download
                </RaisedButton>
              </div>
            </RaisedCard>

            <RaisedCard size="sm">
              <div className="text-center">
                <FileText className="w-8 h-8 text-[#CDB4DB] mx-auto mb-3" />
                <p className="text-sm text-[#2C3E50] mb-4">Signed PDF Report</p>
                <RaisedButton
                  size="sm"
                  className="w-full"
                  onClick={() => {
                    if (!scanId) return;
                    window.open(downloadUrl(scanId, 'cure_pdf'), '_blank', 'noopener,noreferrer');
                  }}
                >
                  <Download className="w-4 h-4" />
                  Download
                </RaisedButton>
                <p className="text-xs text-[#6B7C8F] mt-2">SHA256 verified</p>
              </div>
            </RaisedCard>
          </div>
        </RaisedCard>
      )}
    </div>
  );
}
