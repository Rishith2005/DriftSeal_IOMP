import React, { useRef, useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Database, ChevronDown, ChevronUp } from 'lucide-react';
import { useNavigate } from 'react-router';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';
import { RaisedButton } from '../components/skeuomorphic/RaisedButton';
import { RiskBadge } from '../components/skeuomorphic/RiskBadge';
import { IconContainer } from '../components/skeuomorphic/IconContainer';
import { motion } from 'motion/react';
import { createScan } from '../api/driftsealApi';

interface FileUpload {
  file: File | null;
  status: 'idle' | 'uploaded' | 'error';
  errorText?: string;
  preview?: {
    name: string;
    size: string;
    metrics?: {
      baseline: string;
      primaryValue: string;
      eligible: boolean;
    };
  };
}

export function UploadPage() {
  const navigate = useNavigate();
  const datasetInputRef = useRef<HTMLInputElement>(null);
  const metricsInputRef = useRef<HTMLInputElement>(null);
  const modelInputRef = useRef<HTMLInputElement>(null);
  const [modelFile, setModelFile] = useState<FileUpload>({ file: null, status: 'idle' });
  const [metricsFile, setMetricsFile] = useState<FileUpload>({ file: null, status: 'idle' });
  const [metricsFiles, setMetricsFiles] = useState<File[]>([]);
  const [datasetFile, setDatasetFile] = useState<FileUpload>({ file: null, status: 'idle' });
  const [datasetFiles, setDatasetFiles] = useState<File[]>([]);
  const [datasetExpanded, setDatasetExpanded] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleModelDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith('.py')) {
      setModelFile({
        file,
        status: 'uploaded',
        preview: {
          name: file.name,
          size: formatFileSize(file.size)
        }
      });
    } else {
      setModelFile({ file: null, status: 'error', errorText: 'Please upload a .py file' });
    }
  };

  const handleMetricsDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files || []);
    const accepted = files.filter((f) => {
      const lower = (f?.name || '').toLowerCase();
      return lower.endsWith('.json') || lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg');
    });

    if (accepted.length > 0) {
      setMetricsFiles(accepted);
      const first = accepted[0];
      setMetricsFile({
        file: first,
        status: 'uploaded',
        preview: {
          name: accepted.length === 1 ? first.name : `${first.name} (+${accepted.length - 1} more)`,
          size: formatFileSize(accepted.reduce((sum, f) => sum + f.size, 0)),
          metrics: {
            baseline: '—',
            primaryValue: '—',
            eligible: true
          }
        }
      });
    } else {
      setMetricsFiles([]);
      setMetricsFile({ file: null, status: 'error', errorText: 'Please upload a .json file (e.g. *.meta.json) or an image (.png/.jpg) containing metrics' });
    }
  };

  const handleDatasetDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files || []);
    const accepted = files.filter((f) => {
      const lower = (f?.name || '').toLowerCase();
      return lower.endsWith('.csv') || lower.endsWith('.tsv') || lower.endsWith('.txt') || lower.endsWith('.log') || lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg');
    });

    if (accepted.length > 0) {
      setDatasetFiles(accepted);
      const first = accepted[0];
      setDatasetFile({
        file: first,
        status: 'uploaded',
        preview: {
          name: accepted.length === 1 ? first.name : `${first.name} (+${accepted.length - 1} more)`,
          size: formatFileSize(accepted.reduce((sum, f) => sum + f.size, 0)),
        },
      });
    } else {
      setDatasetFiles([]);
      setDatasetFile({ file: null, status: 'error', errorText: 'Please upload .csv, .tsv, .txt, .log, .png, or .jpg' });
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const handleAnalyze = async () => {
    if (metricsFiles.length === 0 || datasetFiles.length === 0) return;

    setSubmitError(null);
    setProcessing(true);
    setProgress(5);

    const interval = window.setInterval(() => {
      setProgress((p) => (p >= 85 ? p : p + 5));
    }, 250);

    try {
      let lastResult: any = null;
      for (let i = 0; i < metricsFiles.length; i++) {
        const result = await createScan({
          meta: metricsFiles[i],
          dataset: datasetFiles,
          model: modelFile.file,
        });
        lastResult = result;
        const pct = 5 + Math.round(((i + 1) / metricsFiles.length) * 90);
        setProgress((p) => (p < pct ? pct : p));
      }
      window.clearInterval(interval);
      setProgress(100);
      const scanId = lastResult?.record?.scan_id;
      if (scanId) {
        localStorage.setItem('driftseal:last_scan_id', String(scanId));
        navigate(`/dashboard?scan_id=${encodeURIComponent(String(scanId))}`);
      } else {
        setSubmitError('Upload succeeded but no scan_id was returned.');
        setProcessing(false);
        setProgress(0);
      }
    } catch (e: any) {
      window.clearInterval(interval);
      setSubmitError(String(e?.message || e));
      setProcessing(false);
      setProgress(0);
    }
  };

  const canAnalyze = datasetFile.status === 'uploaded' && metricsFile.status === 'uploaded' && datasetFiles.length > 0 && metricsFiles.length > 0;

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Upload Model & Metrics" />
      
      {/* Header */}
      <div className="text-center mb-8">
        <p className="text-[#6B7C8F]">Upload your model files and performance metrics for poisoning detection</p>
      </div>

      {submitError && (
        <RaisedCard>
          <InsetPanel>
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-[#FF8B94] mt-0.5" />
              <div>
                <p className="text-[#2C3E50]">Upload failed</p>
                <p className="text-sm text-[#6B7C8F] break-words">{submitError}</p>
              </div>
            </div>
          </InsetPanel>
        </RaisedCard>
      )}

      {/* Main Upload Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Dataset Upload Card - NOW FIRST */}
        <RaisedCard>
          <div className="mb-4">
            <h2 className="text-[#2C3E50] mb-1">Training Dataset</h2>
            <p className="text-sm text-[#6B7C8F]">Upload training data for analysis</p>
          </div>

          <InsetPanel
            className={`transition-all ${
              datasetFile.status === 'error' 
                ? 'ring-2 ring-[#FF8B94] animate-[shake_0.5s]' 
                : datasetFile.status === 'uploaded'
                ? 'ring-2 ring-[#FFD3B6]'
                : ''
            }`}
            onDragOver={(e: React.DragEvent<HTMLDivElement>) => e.preventDefault()}
            onDrop={handleDatasetDrop}
          >
            <div className="text-center py-12">
              <IconContainer 
                size="lg" 
                variant={datasetFile.status === 'uploaded' ? 'raised' : 'inset'}
                color={datasetFile.status === 'uploaded' ? '#FFD3B6' : undefined}
                className="mx-auto mb-4"
              >
                {datasetFile.status === 'uploaded' ? (
                  <CheckCircle className="w-8 h-8 text-white" />
                ) : datasetFile.status === 'error' ? (
                  <AlertCircle className="w-8 h-8 text-[#FF8B94]" />
                ) : (
                  <Database className="w-8 h-8 text-[#FFD3B6]" />
                )}
              </IconContainer>
              
              {datasetFile.status === 'idle' && (
                <>
                  <h3 className="mb-2 text-[#2C3E50]">Drag & Drop Dataset</h3>
                  <p className="text-sm text-[#6B7C8F] mb-4">or click to browse</p>
                  <div className="flex flex-wrap gap-2 justify-center text-xs text-[#6B7C8F]">
                    <span className="px-3 py-1 bg-white/50 rounded-full">.csv</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.tsv</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.txt</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.log</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.png</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.jpg</span>
                  </div>
                </>
              )}

              {datasetFile.status === 'uploaded' && datasetFile.preview && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-2"
                >
                  <RiskBadge level="warning" label="Uploaded" size="md" />
                  <p className="text-sm text-[#2C3E50] font-medium">{datasetFile.preview.name}</p>
                  <p className="text-xs text-[#6B7C8F]">{datasetFile.preview.size}</p>
                  {datasetFiles.length > 1 && (
                    <div className="pt-2 border-t border-[#E8EDF2] space-y-1 text-left">
                      {datasetFiles.slice(0, 5).map((f) => (
                        <p key={f.name} className="text-xs text-[#6B7C8F] break-all">
                          {f.name}
                        </p>
                      ))}
                      {datasetFiles.length > 5 && <p className="text-xs text-[#6B7C8F]">+{datasetFiles.length - 5} more</p>}
                    </div>
                  )}
                </motion.div>
              )}

              {datasetFile.status === 'error' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <RiskBadge level="critical" label="Invalid Format" size="md" />
                  <p className="text-sm text-[#6B7C8F] mt-2">{datasetFile.errorText || 'Please upload .csv, .tsv, .txt, .log, .png, or .jpg'}</p>
                </motion.div>
              )}
            </div>
          </InsetPanel>

          <input
            type="file"
            accept=".csv,.tsv,.txt,.log,.png,.jpg,.jpeg"
            multiple
            className="hidden"
            ref={datasetInputRef}
            onChange={(e) => {
              const files = e.target.files ? Array.from(e.target.files) : [];
              if (files.length > 0) {
                const fakeEvent = {
                  preventDefault: () => {},
                  dataTransfer: { files }
                } as any;
                handleDatasetDrop(fakeEvent);
              }
            }}
            id="dataset-upload"
          />
          
          {datasetFile.status !== 'uploaded' && (
            <label htmlFor="dataset-upload" className="mt-4 block">
              <RaisedButton
                variant="outline"
                className="w-full cursor-pointer"
                onClick={() => datasetInputRef.current?.click()}
              >
                <Database className="w-4 h-4" />
                Choose File
              </RaisedButton>
            </label>
          )}
        </RaisedCard>

        {/* Metrics JSON Upload Card */}
        <RaisedCard>
          <div className="mb-4">
            <h2 className="text-[#2C3E50] mb-1">Performance Metrics</h2>
            <p className="text-sm text-[#6B7C8F]">Upload JSON metrics file</p>
          </div>

          <InsetPanel
            className={`transition-all ${
              metricsFile.status === 'error' 
                ? 'ring-2 ring-[#FF8B94] animate-[shake_0.5s]' 
                : metricsFile.status === 'uploaded'
                ? 'ring-2 ring-[#A0D8F1]'
                : ''
            }`}
            onDragOver={(e: React.DragEvent<HTMLDivElement>) => e.preventDefault()}
            onDrop={handleMetricsDrop}
          >
            <div className="text-center py-12">
              <IconContainer 
                size="lg" 
                variant={metricsFile.status === 'uploaded' ? 'raised' : 'inset'}
                color={metricsFile.status === 'uploaded' ? '#A0D8F1' : undefined}
                className="mx-auto mb-4"
              >
                {metricsFile.status === 'uploaded' ? (
                  <CheckCircle className="w-8 h-8 text-white" />
                ) : metricsFile.status === 'error' ? (
                  <AlertCircle className="w-8 h-8 text-[#FF8B94]" />
                ) : (
                  <FileText className="w-8 h-8 text-[#CDB4DB]" />
                )}
              </IconContainer>
              
              {metricsFile.status === 'idle' && (
                <>
                  <h3 className="mb-2 text-[#2C3E50]">Drag & Drop Metrics</h3>
                  <p className="text-sm text-[#6B7C8F] mb-4">or click to browse</p>
                  <div className="flex flex-wrap gap-2 justify-center text-xs text-[#6B7C8F]">
                    <span className="px-3 py-1 bg-white/50 rounded-full">.json</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.meta.json</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.png</span>
                    <span className="px-3 py-1 bg-white/50 rounded-full">.jpg</span>
                  </div>
                </>
              )}

              {metricsFile.status === 'uploaded' && metricsFile.preview && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-3"
                >
                  <RiskBadge level="info" label="Schema Valid" size="md" />
                  <p className="text-sm text-[#2C3E50] font-medium">{metricsFile.preview.name}</p>
                  <p className="text-xs text-[#6B7C8F] mb-3">{metricsFile.preview.size}</p>
                  
                  {metricsFile.preview.metrics && (
                    <div className="pt-3 border-t border-[#E8EDF2] space-y-2 text-left">
                      <div className="flex justify-between text-sm">
                        <span className="text-[#6B7C8F]">Baseline</span>
                        <span className="text-[#2C3E50]">{metricsFile.preview.metrics.baseline}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-[#6B7C8F]">Primary Value</span>
                        <span className="text-[#2C3E50]">{metricsFile.preview.metrics.primaryValue}</span>
                      </div>
                      <div className="flex justify-between text-sm items-center">
                        <span className="text-[#6B7C8F]">Eligible</span>
                        <RiskBadge 
                          level={metricsFile.preview.metrics.eligible ? 'clean' : 'warning'} 
                          label={metricsFile.preview.metrics.eligible ? 'Yes' : 'No'} 
                          size="sm" 
                        />
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

              {metricsFile.status === 'error' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <RiskBadge level="critical" label="Invalid JSON" size="md" />
                  <p className="text-sm text-[#6B7C8F] mt-2">{metricsFile.errorText || 'Please upload valid .json file'}</p>
                </motion.div>
              )}
            </div>
          </InsetPanel>

          <input
            type="file"
            accept=".json,.png,.jpg,.jpeg"
            multiple
            className="hidden"
            ref={metricsInputRef}
            onChange={(e) => {
              const files = e.target.files ? Array.from(e.target.files) : [];
              if (files.length > 0) {
                const fakeEvent = {
                  preventDefault: () => {},
                  dataTransfer: { files }
                } as any;
                handleMetricsDrop(fakeEvent);
              }
            }}
            id="metrics-upload"
          />
          
          {metricsFile.status !== 'uploaded' && (
            <label htmlFor="metrics-upload" className="mt-4 block">
              <RaisedButton
                variant="outline"
                className="w-full cursor-pointer"
                onClick={() => metricsInputRef.current?.click()}
              >
                <FileText className="w-4 h-4" />
                Choose File
              </RaisedButton>
            </label>
          )}
        </RaisedCard>
      </div>

      {/* Optional Model File Upload - Collapsible */}
      <RaisedCard>
        <button
          onClick={() => setDatasetExpanded(!datasetExpanded)}
          className="w-full flex items-center justify-between"
        >
          <div className="flex items-center gap-3">
            <IconContainer size="md" variant="inset">
              <Upload className="w-6 h-6 text-[#A0D8F1]" />
            </IconContainer>
            <div className="text-left">
              <h3 className="text-[#2C3E50]">Model File (Optional)</h3>
              <p className="text-sm text-[#6B7C8F]">Upload trained model for enhanced analysis</p>
            </div>
          </div>
          {datasetExpanded ? (
            <ChevronUp className="w-5 h-5 text-[#6B7C8F]" />
          ) : (
            <ChevronDown className="w-5 h-5 text-[#6B7C8F]" />
          )}
        </button>

        {datasetExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="mt-4"
          >
            <InsetPanel
              className={`transition-all ${
                modelFile.status === 'uploaded' ? 'ring-2 ring-[#A0D8F1]' : ''
              }`}
              onDragOver={(e: React.DragEvent<HTMLDivElement>) => e.preventDefault()}
              onDrop={handleModelDrop}
            >
              <div className="text-center py-8">
                <IconContainer 
                  size="md" 
                  variant={modelFile.status === 'uploaded' ? 'raised' : 'inset'}
                  color={modelFile.status === 'uploaded' ? '#A0D8F1' : undefined}
                  className="mx-auto mb-4"
                >
                  {modelFile.status === 'uploaded' ? (
                    <CheckCircle className="w-6 h-6 text-white" />
                  ) : (
                    <Upload className="w-6 h-6 text-[#A0D8F1]" />
                  )}
                </IconContainer>
                
                {modelFile.status === 'idle' && (
                  <>
                    <p className="text-sm text-[#6B7C8F] mb-3">Drag & drop model or click to browse</p>
                    <div className="flex flex-wrap gap-2 justify-center text-xs text-[#6B7C8F]">
                      <span className="px-3 py-1 bg-white/50 rounded-full">.py</span>
                    </div>
                  </>
                )}

                {modelFile.status === 'uploaded' && modelFile.preview && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-2"
                  >
                    <RiskBadge level="info" label="Uploaded" size="sm" />
                    <p className="text-sm text-[#2C3E50]">{modelFile.preview.name}</p>
                    <p className="text-xs text-[#6B7C8F]">{modelFile.preview.size}</p>
                  </motion.div>
                )}

                {modelFile.status === 'error' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <RiskBadge level="critical" label="Invalid Format" size="sm" />
                    <p className="text-sm text-[#6B7C8F] mt-2">{modelFile.errorText || 'Please upload a .py file'}</p>
                  </motion.div>
                )}
              </div>
            </InsetPanel>

            <input
              type="file"
              accept=".py"
              className="hidden"
              ref={modelInputRef}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  const fakeEvent = {
                    preventDefault: () => {},
                    dataTransfer: { files: [file] }
                  } as any;
                  handleModelDrop(fakeEvent);
                }
              }}
              id="model-upload"
            />
            
            {modelFile.status !== 'uploaded' && (
              <label htmlFor="model-upload" className="mt-4 block">
                <RaisedButton
                  variant="outline"
                  className="w-full cursor-pointer"
                  onClick={() => modelInputRef.current?.click()}
                >
                  <Upload className="w-4 h-4" />
                  Choose File
                </RaisedButton>
              </label>
            )}
          </motion.div>
        )}
      </RaisedCard>

      {/* Upload CTA Section */}
      <RaisedCard>
        <div className="text-center">
          <RaisedButton
            variant="primary"
            size="lg"
            onClick={handleAnalyze}
            disabled={!canAnalyze || processing}
            className="mb-4"
          >
            {processing ? (
              <>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                >
                  <Upload className="w-5 h-5" />
                </motion.div>
                Analyzing...
              </>
            ) : (
              <>
                <CheckCircle className="w-5 h-5" />
                Analyze & Generate Verdict
              </>
            )}
          </RaisedButton>

          {processing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <InsetPanel size="sm" className="max-w-md mx-auto">
                <div className="space-y-3">
                  <div className="relative h-4 rounded-full bg-[#F0F4F8] overflow-hidden"
                    style={{ boxShadow: 'var(--shadow-soft-inset)' }}
                  >
                    <motion.div
                      className="h-full rounded-full bg-gradient-to-r from-[#A0D8F1] to-[#A8E6CF]"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <p className="text-sm text-[#6B7C8F]">
                    {progress < 30 && 'Validating files...'}
                    {progress >= 30 && progress < 60 && 'Extracting metrics...'}
                    {progress >= 60 && progress < 90 && 'Running drift detection...'}
                    {progress >= 90 && 'Generating verdict...'}
                  </p>
                </div>
              </InsetPanel>
            </motion.div>
          )}

          {!canAnalyze && !processing && (
            <p className="text-sm text-[#6B7C8F]">
              Please upload both training dataset and metrics files to continue
            </p>
          )}
        </div>
      </RaisedCard>
    </div>
  );
}
