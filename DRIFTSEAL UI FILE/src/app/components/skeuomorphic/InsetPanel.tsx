import React from 'react';
import { cn } from '../ui/utils';

interface InsetPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg';
}

export function InsetPanel({ children, className, size = 'md', style, ...rest }: InsetPanelProps) {
  const sizeStyles = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };

  return (
    <div
      {...rest}
      className={cn("rounded-2xl bg-[#f0f6f8] bg-[#f0f5f8] bg-[#f0f4f8] bg-[#f0f3f8] bg-[#f0f2f8] bg-[#f0f1f8] bg-[#f0f0f8]", sizeStyles[size], className)}
      style={{
        boxShadow: 'var(--shadow-soft-inset-lg)',
        ...(style || {})
      }}
    >
      {children}
    </div>
  );
}
