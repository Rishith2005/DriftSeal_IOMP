export type Severity = 'info' | 'low' | 'medium' | 'critical'

export type ScanRecord = {
  scan_id: string
  created_at: string
  model_name: string
  domain?: string
  meta_sha256?: string | null
  meta_path?: string
  poison_score?: number
  threshold?: number
  poisoned?: boolean
  drift_score?: number
  verdict?: string
  confidence?: string
  detector_model_dir?: string
  autoencoder_available?: boolean
}

export type CreateScanResponse = {
  record: ScanRecord
  score: {
    raw_scores: Record<string, number>
    fingerprint_source: string
  }
  performance_metrics?: any
  uploads?: any
}

export type VerifyResponse = {
  record: ScanRecord
  verification: any
}

export type CureResponse = {
  record: ScanRecord
  cure: any
  cure_artifacts?: any
  verification?: any
}

export type ListScansResponse = {
  items: ScanRecord[]
  total?: number
}

export type MonitoringSummaryResponse = {
  current: {
    drift_score: number
    poison_score?: number | null
    updated_at: string
  }
  series: Array<{ t: string; drift_score: number; poison_score?: number | null }>
  alerts: Array<{ id: string; t: string; severity: Severity; message: string; metric?: string }>
}

export type HealthResponse = {
  ok: true
  model_dir?: string | null
  threshold?: number | null
  autoencoder_available?: boolean | null
}

export type RemediationFixRecommendation = {
  id: number
  name: string
  type: string
  metrics: {
    expected_f1_boost_pct: number
    gpu_minutes: number
    success_rate_pct: number
  }
  params: { dataset_contamination: number; predictions_contamination: number; retrain: boolean; fix_id: number }
}

export type RemediationRecommendationsResponse = {
  scan_id: string
  poison_score: number
  severity_pct: number
  dataset_rows?: number | null
  default_fix_id: number
  fixes: RemediationFixRecommendation[]
}

async function jsonFetch<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init)
  const text = await res.text()
  let data: any = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = { raw: text }
  }
  if (!res.ok) {
    const msg = (data && (data.error || data.message)) || `Request failed (${res.status})`
    throw new Error(msg)
  }
  return data as T
}

export async function createScan(params: {
  meta: File
  predictions?: File | null
  samples?: File | null
  dataset?: File | File[] | null
  model?: File | null
}): Promise<CreateScanResponse> {
  const form = new FormData()
  form.append('meta', params.meta)
  if (params.predictions) form.append('predictions', params.predictions)
  if (params.samples) form.append('samples', params.samples)
  if (Array.isArray(params.dataset)) {
    for (const f of params.dataset) form.append('dataset', f)
  } else if (params.dataset) {
    form.append('dataset', params.dataset)
  }
  if (params.model) form.append('model', params.model)
  return jsonFetch<CreateScanResponse>('/api/scans', { method: 'POST', body: form })
}

export async function getScan(scanId: string): Promise<any> {
  return jsonFetch<any>(`/api/scans/${encodeURIComponent(scanId)}`)
}

export async function getScanRecommendations(scanId: string): Promise<RemediationRecommendationsResponse> {
  return jsonFetch<RemediationRecommendationsResponse>(`/api/scans/${encodeURIComponent(scanId)}/recommendations`)
}

export async function listScans(params?: { limit?: number; offset?: number }): Promise<ListScansResponse> {
  const q = new URLSearchParams()
  if (params?.limit != null) q.set('limit', String(params.limit))
  if (params?.offset != null) q.set('offset', String(params.offset))
  const qs = q.toString()
  return jsonFetch<ListScansResponse>(`/api/scans${qs ? `?${qs}` : ''}`)
}

export async function verifyScan(scanId: string, body?: { prediction_contamination?: number; canary_n?: number }): Promise<VerifyResponse> {
  return jsonFetch<VerifyResponse>(`/api/scans/${encodeURIComponent(scanId)}/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
}

export async function cureScan(
  scanId: string,
  body?: { dataset_contamination?: number; predictions_contamination?: number; retrain?: boolean; fix_id?: number; force?: boolean },
): Promise<CureResponse> {
  return jsonFetch<CureResponse>(`/api/scans/${encodeURIComponent(scanId)}/cure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
}

export type DownloadKind =
  | 'meta'
  | 'predictions'
  | 'dataset'
  | 'cure_report'
  | 'cure_pdf'
  | 'cure_bundle'
  | 'model'
  | 'cured_model'
  | 'cured_metrics'

export function downloadUrl(scanId: string, kind: DownloadKind): string {
  const q = new URLSearchParams({ kind })
  return `/api/scans/${encodeURIComponent(scanId)}/download?${q.toString()}`
}

export async function getMonitoringSummary(): Promise<MonitoringSummaryResponse> {
  return jsonFetch<MonitoringSummaryResponse>('/api/monitoring/summary')
}

export async function resetMonitoring(): Promise<{ ok: true }> {
  return jsonFetch<{ ok: true }>('/api/monitoring/reset', { method: 'POST' })
}

export async function getHealth(): Promise<HealthResponse> {
  return jsonFetch<HealthResponse>('/api/health')
}
