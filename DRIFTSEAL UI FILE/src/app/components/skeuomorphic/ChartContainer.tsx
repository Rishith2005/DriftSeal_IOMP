import React from 'react';
import { cn } from '../ui/utils';
import { InsetPanel } from './InsetPanel';

interface ChartContainerProps {
  children: React.ReactNode;
  title?: string;
  description?: string;
  aspectRatio?: '16/9' | '4/3' | '1/1';
  className?: string;
}

export function ChartContainer({ 
  children, 
  title, 
  description,
  aspectRatio = '16/9',
  className 
}: ChartContainerProps) {
  return (
    <div className={className}>
      {(title || description) && (
        <div className="mb-4">
          {title && <h3 className="text-[#2C3E50] mb-1">{title}</h3>}
          {description && <p className="text-sm text-[#6B7C8F]">{description}</p>}
        </div>
      )}
      <InsetPanel>
        <div 
          className={cn("w-full overflow-hidden")}
          style={{ aspectRatio }}
        >
          {children}
        </div>
      </InsetPanel>
    </div>
  );
}
