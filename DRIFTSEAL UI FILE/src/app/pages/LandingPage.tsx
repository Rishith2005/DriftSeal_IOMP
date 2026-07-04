import React from 'react';
import { useNavigate } from 'react-router';
import { Shield, Activity, LineChart, Zap, Github, FileText, Layers } from 'lucide-react';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { RaisedButton } from '../components/skeuomorphic/RaisedButton';

export function LandingPage() {
  const navigate = useNavigate();

  const features = [
    {
      icon: <LineChart className="w-8 h-8" />,
      title: 'Metrics Drift Visualization',
      description: 'Real-time visualization of behavioral drift across your ML models with interactive heatmaps.'
    },
    {
      icon: <Activity className="w-8 h-8" />,
      title: 'Threshold-Based Poison Score',
      description: 'Advanced scoring system that detects anomalies through statistical analysis.'
    },
    {
      icon: <Layers className="w-8 h-8" />,
      title: 'Interactive Heatmaps',
      description: 'Explore metrics across test types with zoom, filter, and detailed hover information.'
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: 'One-Click Dataset Sanitization',
      description: 'Automated remediation workflows to cure poisoned models instantly.'
    }
  ];

  const techStack = ['FastAPI', 'React', 'Plotly', 'Docker'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#F7F9FB] via-[#F0F4F8] to-[#E8EDF2]">
      {/* Top Navigation */}
      <div className="container mx-auto px-6 py-6">
        <div className="flex items-center justify-between">
          <button 
            onClick={() => navigate('/')}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="w-10 h-10 rounded-2xl flex items-center justify-center"
              style={{
                backgroundColor: '#A0D8F1',
                boxShadow: 'var(--shadow-soft-outer)'
              }}
            >
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-[#2C3E50]">Drift Seal</h3>
              <p className="text-xs text-[#6B7C8F]">ML Security Platform</p>
            </div>
          </button>

          <RaisedButton 
            onClick={() => navigate('/dashboard')}
            variant="primary"
          >
            <Activity className="w-4 h-4" />
            Open Dashboard
          </RaisedButton>
        </div>
      </div>

      {/* Hero Section */}
      <div className="container mx-auto px-6 py-16 text-center">
        <div className="text-center mb-16">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Shield className="w-12 h-12 text-[#A0D8F1]" />
            <h1 className="text-5xl">Drift Seal</h1>
          </div>
          <p className="text-3xl mb-4 text-[#2C3E50]">
            Detect & Cure Poisoned ML Instantly
          </p>
          <p className="text-xl text-[#6B7C8F] max-w-2xl mx-auto mb-12">
            Metrics-first ML poisoning detection with advanced behavioral drift analysis 
            and automated remediation workflows.
          </p>

          {/* CTA Buttons */}
          <div className="flex gap-4 justify-center mb-16">
            <RaisedButton
              variant="primary"
              size="lg"
              onClick={() => navigate('/upload')}
            >
              <Activity className="w-5 h-5" />
              Analyze Metrics Now
            </RaisedButton>
            <RaisedButton
              variant="outline"
              size="lg"
              onClick={() => navigate('/upload')}
            >
              <Shield className="w-5 h-5" />
              View Demo
            </RaisedButton>
          </div>

          {/* Demo Mockup */}
          <RaisedCard className="max-w-4xl mx-auto">
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl text-[#2C3E50]">Quick Analysis Preview</h3>
                <div className="flex gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#FF8B94]" />
                  <div className="w-3 h-3 rounded-full bg-[#FFD3B6]" />
                  <div className="w-3 h-3 rounded-full bg-[#A8E6CF]" />
                </div>
              </div>
              
              {/* Mock visualization */}
              <div className="grid grid-cols-3 gap-4">
                <div className="p-6 bg-gradient-to-br from-[#A0D8F1] to-[#CDB4DB] rounded-2xl text-white">
                  <p className="text-sm opacity-80 mb-2">Upload JSON</p>
                  <FileText className="w-8 h-8" />
                </div>
                <div className="p-6 bg-gradient-to-br from-[#FFD3B6] to-[#FF8B94] rounded-2xl text-white">
                  <p className="text-sm opacity-80 mb-2">Analyze Drift</p>
                  <Activity className="w-8 h-8" />
                </div>
                <div className="p-6 bg-gradient-to-br from-[#A8E6CF] to-[#A0D8F1] rounded-2xl text-white">
                  <p className="text-sm opacity-80 mb-2">Clean Verdict</p>
                  <Shield className="w-8 h-8" />
                </div>
              </div>
            </div>
          </RaisedCard>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-20">
          {features.map((feature, index) => (
            <RaisedCard key={index} size="md">
              <div className="text-[#A0D8F1] mb-4">
                {feature.icon}
              </div>
              <h3 className="mb-3 text-[#2C3E50]">{feature.title}</h3>
              <p className="text-[#6B7C8F]">{feature.description}</p>
            </RaisedCard>
          ))}
        </div>

        {/* Tech Badges */}
        <div className="text-center mb-12">
          <p className="text-sm text-[#6B7C8F] mb-4">Powered by</p>
          <div className="flex gap-3 justify-center flex-wrap">
            {techStack.map((tech) => (
              <div
                key={tech}
                className="px-6 py-3 bg-white rounded-full text-[#2C3E50]"
                style={{
                  boxShadow: 'var(--shadow-soft-outer)'
                }}
              >
                {tech}
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center space-y-4">
          <div className="flex gap-6 justify-center text-[#6B7C8F]">
            <button className="hover:text-[#A0D8F1] transition-colors flex items-center gap-2">
              <Github className="w-5 h-5" />
              GitHub
            </button>
            <button className="hover:text-[#A0D8F1] transition-colors flex items-center gap-2">
              <Layers className="w-5 h-5" />
              HF Spaces
            </button>
            <button className="hover:text-[#A0D8F1] transition-colors flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Documentation
            </button>
          </div>
          <p className="text-sm text-[#6B7C8F]">
            Multi-domain Support • Enterprise-grade Security • Open Source
          </p>
        </footer>
      </div>
    </div>
  );
}
