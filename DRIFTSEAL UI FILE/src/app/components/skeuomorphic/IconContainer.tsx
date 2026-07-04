import React from 'react';
import { cn } from '../ui/utils';

interface IconContainerProps {
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'raised' | 'inset' | 'flat';
  color?: string;
  className?: string;
}

export function IconContainer({ 
  children, 
  size = 'md', 
  variant = 'flat',
  color,
  className 
}: IconContainerProps) {
  const sizeStyles = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  };

  const variantStyles = {
    raised: 'bg-white',
    inset: 'bg-[#F0F4F8]',
    flat: 'bg-transparent'
  };

  return (
    <div
      className={cn(
        "rounded-2xl flex items-center justify-center flex-shrink-0 overflow-hidden",
        sizeStyles[size],
        variantStyles[variant],
        className
      )}
      style={{
        boxShadow: variant === 'raised' ? 'var(--shadow-soft-outer)' : variant === 'inset' ? 'var(--shadow-soft-inset)' : 'none',
        backgroundColor: color || undefined
      }}
    >
      {children}
    </div>
  );
}
