import React from 'react';
import { cn } from '../ui/utils';

interface InsetInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export function InsetInput({ label, error, className, ...props }: InsetInputProps) {
  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm text-[#2C3E50]">
          {label}
        </label>
      )}
      <input
        className={cn(
          "w-full px-4 py-3 rounded-2xl bg-[#F0F4F8] border-0",
          "focus:outline-none focus:ring-2 focus:ring-[#A0D8F1]",
          "placeholder:text-[#6B7C8F]",
          className
        )}
        style={{
          boxShadow: 'var(--shadow-soft-inset)'
        }}
        {...props}
      />
      {error && (
        <p className="text-sm text-[#FF8B94]">{error}</p>
      )}
    </div>
  );
}
