import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, TrendingUp, AlertCircle } from 'lucide-react';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';
import { RiskBadge } from '../components/skeuomorphic/RiskBadge';
import { getMonitoringSummary } from '../api/driftsealApi';

export function MonitoringPage() {
  const [summary, setSummary] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = useCallback(async () => {
    setError(null);
    const s = await getMonitoringSummary();
    setSummary(s);
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      try {
        if (!cancelled) await fetchSummary();
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
      }
    }

    run();
    const id = window.setInterval(run, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [fetchSummary]);

  const driftData = useMemo(() => {
    const series = summary?.series || [];
    return series.map((p: any) => ({
      timestamp: String(p.t || '').slice(11, 16) || String(p.t || ''),
      driftScore: Math.round((Number(p.drift_score) || 0) * 100),
      poisonScore: p.poison_score == null ? null : Math.round((Number(p.poison_score) || 0) * 100),
    }));
  }, [summary]);

  const currentPoisonPct = Math.round((Number(summary?.current?.poison_score) || 0) * 100);
  const currentDriftPct = Math.round((Number(summary?.current?.drift_score) || 0) * 100);
  const severity: 'info' | 'low' | 'medium' | 'critical' = currentPoisonPct >= 75 ? 'critical' : currentPoisonPct >= 50 ? 'medium' : currentPoisonPct >= 25 ? 'low' : 'info';

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Real-Time Monitoring" />

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <RaisedCard>
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm text-[#6B7C8F] mb-1">Poison Score</p>
              <p className="text-3xl text-[#2C3E50]">{currentPoisonPct}%</p>
            </div>
            <Activity className="w-8 h-8 text-[#FFD3B6]" />
          </div>
          <div className="flex items-center gap-2 text-sm">
            <TrendingUp className="w-4 h-4 text-[#FF8B94] rotate-180" />
            <span className="text-[#6B7C8F]">Latest update: {summary?.current?.updated_at || '—'}</span>
          </div>
        </RaisedCard>

        <RaisedCard>
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm text-[#6B7C8F] mb-1">Drift Score</p>
              <p className="text-3xl text-[#FF8B94]">{currentDriftPct}%</p>
            </div>
            <AlertCircle className="w-8 h-8 text-[#FF8B94]" />
          </div>
          <div className="flex items-center gap-2 text-sm">
            <RiskBadge level={severity === 'critical' ? 'critical' : severity === 'medium' ? 'medium' : severity === 'low' ? 'low' : 'info'} size="sm" label={severity === 'critical' ? 'Critical' : severity === 'medium' ? 'Elevated' : severity === 'low' ? 'Low' : 'Info'} />
          </div>
        </RaisedCard>

        <RaisedCard>
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-sm text-[#6B7C8F] mb-1">Uptime</p>
              <p className="text-3xl text-[#A8E6CF]">99.8%</p>
            </div>
            <Activity className="w-8 h-8 text-[#A8E6CF]" />
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-[#6B7C8F]">Last 30 days</span>
          </div>
        </RaisedCard>
      </div>

      {/* Time-Series Graphs */}
      <RaisedCard>
        <h2 className="mb-4 text-[#2C3E50]">Drift Over Time</h2>
        {error && <p className="text-sm text-[#FF8B94] mb-3">{error}</p>}
        <InsetPanel>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={driftData}>
              <defs>
                <linearGradient id="poisonGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#FF8B94" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#FF8B94" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#E8EDF2" />
              <XAxis dataKey="timestamp" stroke="#6B7C8F" />
              <YAxis stroke="#6B7C8F" />
              <Tooltip 
                contentStyle={{ 
                  background: 'white', 
                  border: 'none', 
                  borderRadius: '12px',
                  boxShadow: 'var(--shadow-soft-outer)'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="poisonScore" 
                stroke="#FF8B94" 
                strokeWidth={3}
                fill="url(#poisonGradient)" 
              />
            </AreaChart>
          </ResponsiveContainer>
        </InsetPanel>
      </RaisedCard>

      <RaisedCard>
        <h2 className="mb-4 text-[#2C3E50]">Performance Metrics</h2>
        <InsetPanel>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={driftData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E8EDF2" />
              <XAxis dataKey="timestamp" stroke="#6B7C8F" />
              <YAxis stroke="#6B7C8F" domain={[0, 100]} />
              <Tooltip 
                contentStyle={{ 
                  background: 'white', 
                  border: 'none', 
                  borderRadius: '12px',
                  boxShadow: 'var(--shadow-soft-outer)'
                }}
              />
              <Line type="monotone" dataKey="driftScore" stroke="#A0D8F1" strokeWidth={3} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </InsetPanel>
      </RaisedCard>

      {/* Alert Feed */}
      <RaisedCard>
        <h2 className="mb-4 text-[#2C3E50]">Recent Alerts</h2>
        <div className="space-y-3">
          {(summary?.alerts || []).map((alert: any, index: number) => (
            <InsetPanel key={index} size="sm">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 text-sm text-[#6B7C8F] w-16">
                  {String(alert.t || '').slice(11, 16) || alert.t}
                </div>
                <div className="flex-1">
                  <p className="text-[#2C3E50] mb-1">{alert.message}</p>
                  <p className="text-sm text-[#6B7C8F]">{alert.metric}</p>
                </div>
                <RiskBadge level={alert.severity} size="sm" />
              </div>
            </InsetPanel>
          ))}
        </div>
      </RaisedCard>
    </div>
  );
}
