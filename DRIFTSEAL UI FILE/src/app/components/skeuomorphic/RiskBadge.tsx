import React from 'react';
import { cn } from '../ui/utils';

interface RiskBadgeProps {
  level: 'clean' | 'low' | 'medium' | 'critical' | 'info' | 'warning';
  label?: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function RiskBadge({ level, label, className, size = 'md' }: RiskBadgeProps) {
  const sizeStyles = {
    sm: 'px-2.5 py-1 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base'
  };

  const levelConfig = {
    clean: {
      bg: '#A8E6CF',
      text: '#2D6A4F',
      label: label || 'Clean'
    },
    low: {
      bg: '#A0D8F1',
      text: '#1A4D6E',
      label: label || 'Low Risk'
    },
    medium: {
      bg: '#FFD3B6',
      text: '#8B5A3C',
      label: label || 'Medium Risk'
    },
    critical: {
      bg: '#FF8B94',
      text: '#8B3A3F',
      label: label || 'Critical'
    },
    info: {
      bg: '#A0D8F1',
      text: '#1A4D6E',
      label: label || 'Info'
    },
    warning: {
      bg: '#FFD3B6',
      text: '#8B5A3C',
      label: label || 'Warning'
    }
  };

  const config = levelConfig[level];

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full relative overflow-hidden",
        sizeStyles[size],
        className
      )}
      style={{
        backgroundColor: config.bg,
        color: config.text,
        boxShadow: 'var(--shadow-soft-outer)'
      }}
    >
      <div 
        className="absolute inset-0 pointer-events-none opacity-20"
        style={{
          background: 'var(--gradient-glossy)'
        }}
      />
      <span className="relative z-10">{config.label}</span>
    </span>
  );
}
