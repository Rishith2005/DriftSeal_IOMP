import React from 'react';
import { cn } from '../ui/utils';

interface GaugeWidgetProps {
  value: number; // 0-100
  level: 'clean' | 'low' | 'medium' | 'critical';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function GaugeWidget({ value, level, label, size = 'md', className }: GaugeWidgetProps) {
  const sizeConfig = {
    sm: { diameter: 120, strokeWidth: 12 },
    md: { diameter: 180, strokeWidth: 16 },
    lg: { diameter: 240, strokeWidth: 20 }
  };

  const config = sizeConfig[size];
  const radius = (config.diameter - config.strokeWidth) / 2;
  const circumference = radius * Math.PI * 1.5; // 270 degrees
  const offset = circumference - (value / 100) * circumference;

  const levelColors = {
    clean: '#A8E6CF',
    low: '#A0D8F1',
    medium: '#FFD3B6',
    critical: '#FF8B94'
  };

  const levelLabels = {
    clean: 'Clean',
    low: 'Low Risk',
    medium: 'Medium Risk',
    critical: 'Critical'
  };

  return (
    <div className={cn("flex flex-col items-center", className)}>
      <div 
        className="relative rounded-full bg-white p-4"
        style={{
          width: config.diameter + 32,
          height: config.diameter + 32,
          boxShadow: 'var(--shadow-soft-outer-lg)'
        }}
      >
        {/* Glossy overlay */}
        <div 
          className="absolute inset-0 pointer-events-none opacity-30 rounded-full"
          style={{
            background: 'var(--gradient-glossy)'
          }}
        />
        
        <svg
          width={config.diameter}
          height={config.diameter}
          className="relative z-10"
        >
          {/* Background arc */}
          <circle
            cx={config.diameter / 2}
            cy={config.diameter / 2}
            r={radius}
            fill="none"
            stroke="#E8EDF2"
            strokeWidth={config.strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={0}
            transform={`rotate(135 ${config.diameter / 2} ${config.diameter / 2})`}
            strokeLinecap="round"
          />
          
          {/* Value arc */}
          <circle
            cx={config.diameter / 2}
            cy={config.diameter / 2}
            r={radius}
            fill="none"
            stroke={levelColors[level]}
            strokeWidth={config.strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={offset}
            transform={`rotate(135 ${config.diameter / 2} ${config.diameter / 2})`}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.5s ease' }}
          />
          
          {/* Center text */}
          <text
            x="50%"
            y="45%"
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-[#2C3E50]"
            style={{ fontSize: size === 'lg' ? '36px' : size === 'md' ? '28px' : '20px', fontWeight: 500 }}
          >
            {value}%
          </text>
          <text
            x="50%"
            y="60%"
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-[#6B7C8F]"
            style={{ fontSize: size === 'lg' ? '16px' : size === 'md' ? '14px' : '12px' }}
          >
            {levelLabels[level]}
          </text>
        </svg>
      </div>
      {label && (
        <p className="mt-4 text-center text-[#2C3E50]">{label}</p>
      )}
    </div>
  );
}
