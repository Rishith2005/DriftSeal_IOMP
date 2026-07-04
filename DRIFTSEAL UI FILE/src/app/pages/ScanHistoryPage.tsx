import React, { useEffect, useState } from 'react';
import { Calendar, FileText, Download } from 'lucide-react';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';
import { RiskBadge } from '../components/skeuomorphic/RiskBadge';
import { RaisedButton } from '../components/skeuomorphic/RaisedButton';
import { useNavigate } from 'react-router';
import { downloadUrl, listScans } from '../api/driftsealApi';

export function ScanHistoryPage() {
  const navigate = useNavigate();
  const [items, setItems] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      setError(null);
      try {
        const res = await listScans({ limit: 50, offset: 0 });
        if (!cancelled) setItems(res.items || []);
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message || e));
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, []);

  const toRisk = (r: any): 'clean' | 'low' | 'medium' | 'critical' => {
    const poisoned = Boolean(r?.poisoned);
    const ps = Number.isFinite(r?.poison_score) ? Number(r.poison_score) : 0;
    const th = Number.isFinite(r?.threshold) ? Number(r.threshold) : 0.5;
    if (poisoned || ps > th) return 'critical';
    if (ps >= th * 0.9) return 'medium';
    if (ps >= th * 0.75) return 'low';
    return 'clean';
  };

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Scan History" />

      {/* Scan List */}
      <div className="grid grid-cols-1 gap-4">
        {error && (
          <RaisedCard>
            <InsetPanel size="sm">
              <p className="text-sm text-[#FF8B94]">{error}</p>
            </InsetPanel>
          </RaisedCard>
        )}
        {items.map((scan) => (
          <RaisedCard
            key={scan.scan_id}
            className="hover:scale-[1.01] transition-transform cursor-pointer"
            onClick={() => {
              localStorage.setItem('driftseal:last_scan_id', String(scan.scan_id));
              navigate(`/dashboard?scan_id=${encodeURIComponent(String(scan.scan_id))}`);
            }}
          >
            <div className="flex items-center gap-6">
              {/* Icon */}
              <div 
                className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0"
                style={{
                  backgroundColor: '#F0F4F8',
                  boxShadow: 'var(--shadow-soft-inset)'
                }}
              >
                <FileText className="w-8 h-8 text-[#A0D8F1]" />
              </div>

              {/* Info */}
              <div className="flex-1 min-w-0">
                <h3 className="text-[#2C3E50] mb-1 truncate">{scan.model_name || scan.scan_id}</h3>
                <div className="flex items-center gap-4 text-sm text-[#6B7C8F]">
                  <div className="flex items-center gap-1">
                    <Calendar className="w-4 h-4" />
                    <span>{scan.created_at || '—'}</span>
                  </div>
                  <span>•</span>
                  <span>{scan.domain || '—'}</span>
                </div>
              </div>

              {/* Metrics */}
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <p className="text-sm text-[#6B7C8F] mb-1">Poison Score</p>
                  <p className="text-xl text-[#2C3E50]">{Math.round((Number(scan.poison_score) || 0) * 100)}%</p>
                </div>

                <RiskBadge level={toRisk(scan)} size="lg" />

                <RaisedButton
                  variant="outline"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(downloadUrl(String(scan.scan_id), 'meta'), '_blank', 'noopener,noreferrer');
                  }}
                >
                  <Download className="w-4 h-4" />
                  Export
                </RaisedButton>
              </div>
            </div>
          </RaisedCard>
        ))}
      </div>
    </div>
  );
}
