import React from 'react';
import { cn } from '../ui/utils';

interface RaisedButtonProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
}

export function RaisedButton({ 
  children, 
  className, 
  variant = 'primary', 
  size = 'md',
  onClick,
  disabled,
  type = 'button'
}: RaisedButtonProps) {
  const sizeStyles = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3',
    lg: 'px-8 py-4'
  };

  const variantStyles = {
    primary: 'bg-[#A0D8F1] text-[#1A4D6E]',
    secondary: 'bg-[#CDB4DB] text-[#4A3F5C]',
    outline: 'bg-white border-2 border-[#A0D8F1] text-[#1A4D6E]'
  };

  return (
    <button
      type={type}
      className={cn(
        "rounded-2xl relative overflow-hidden transition-all",
        "hover:brightness-105 active:scale-95",
        "disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100",
        sizeStyles[size],
        variantStyles[variant],
        className
      )}
      style={{
        boxShadow: disabled ? 'none' : 'var(--shadow-soft-outer)'
      }}
      onClick={onClick}
      disabled={disabled}
    >
      {/* Glossy overlay */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-40"
        style={{
          background: 'var(--gradient-glossy)'
        }}
      />
      <span className="relative z-10 flex items-center justify-center gap-2">
        {children}
      </span>
    </button>
  );
}
