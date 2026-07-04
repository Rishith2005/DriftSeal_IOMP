import React from 'react';
import { cn } from '../ui/utils';

interface SoftSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  showValue?: boolean;
  className?: string;
  unit?: string;
}

export function SoftSlider({ 
  value, 
  onChange, 
  min = 0, 
  max = 100, 
  step = 1, 
  label,
  showValue = true,
  className,
  unit = ''
}: SoftSliderProps) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn("space-y-2", className)}>
      {(label || showValue) && (
        <div className="flex justify-between items-center">
          {label && <span className="text-sm text-[#2C3E50]">{label}</span>}
          {showValue && <span className="text-sm text-[#6B7C8F]">{value}{unit}</span>}
        </div>
      )}
      <div className="relative">
        <div 
          className="h-3 rounded-full bg-[#F0F4F8]"
          style={{
            boxShadow: 'var(--shadow-soft-inset)'
          }}
        >
          <div 
            className="h-full rounded-full bg-gradient-to-r from-[#A0D8F1] to-[#CDB4DB]"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full opacity-0 cursor-pointer"
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-5 h-5 rounded-full bg-white pointer-events-none"
          style={{
            left: `calc(${percentage}% - 10px)`,
            boxShadow: 'var(--shadow-soft-outer)'
          }}
        />
      </div>
    </div>
  );
}
