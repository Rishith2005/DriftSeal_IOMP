import React from 'react';
import { cn } from '../ui/utils';

interface SoftToggleProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label?: string;
  className?: string;
}

export function SoftToggle({ checked, onCheckedChange, label, className }: SoftToggleProps) {
  return (
    <label className={cn("flex items-center gap-3 cursor-pointer", className)}>
      <div
        className={cn(
          "relative w-14 h-8 rounded-full transition-colors",
          checked ? "bg-[#A8E6CF]" : "bg-[#E8EDF2]"
        )}
        style={{
          boxShadow: 'var(--shadow-soft-inset)'
        }}
        onClick={() => onCheckedChange(!checked)}
      >
        <div
          className={cn(
            "absolute top-1 w-6 h-6 rounded-full bg-white transition-transform",
            checked ? "translate-x-7" : "translate-x-1"
          )}
          style={{
            boxShadow: 'var(--shadow-soft-outer)'
          }}
        />
      </div>
      {label && <span className="text-[#2C3E50]">{label}</span>}
    </label>
  );
}
