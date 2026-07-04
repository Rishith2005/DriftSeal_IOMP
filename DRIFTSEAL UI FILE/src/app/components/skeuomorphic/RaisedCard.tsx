import React from 'react';
import { cn } from '../ui/utils';

interface RaisedCardProps {
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  onClick?: () => void;
}

export function RaisedCard({ children, className, size = 'md', onClick }: RaisedCardProps) {
  const sizeStyles = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
    xl: 'p-10'
  };

  return (
    <div
      className={cn(
        "bg-white rounded-3xl relative overflow-hidden",
        sizeStyles[size],
        onClick && "cursor-pointer transition-transform hover:scale-[1.02]",
        className
      )}
      style={{
        boxShadow: 'var(--shadow-soft-outer-lg)'
      }}
      onClick={onClick}
    >
      {/* Glossy gradient overlay */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-30 rounded-3xl"
        style={{
          background: 'var(--gradient-glossy)'
        }}
      />
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
}