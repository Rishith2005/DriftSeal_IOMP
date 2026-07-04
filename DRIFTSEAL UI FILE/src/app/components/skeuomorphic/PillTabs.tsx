import React from 'react';
import { cn } from '../ui/utils';

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
}

interface PillTabsProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  className?: string;
}

export function PillTabs({ tabs, activeTab, onTabChange, className }: PillTabsProps) {
  return (
    <div 
      className={cn("inline-flex gap-2 p-2 rounded-3xl bg-[#F0F4F8]", className)}
      style={{
        boxShadow: 'var(--shadow-soft-inset)'
      }}
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={cn(
            "px-6 py-2.5 rounded-2xl transition-all flex items-center gap-2",
            activeTab === tab.id
              ? "bg-[#A0D8F1] text-[#1A4D6E]"
              : "bg-transparent text-[#6B7C8F] hover:text-[#2C3E50]"
          )}
          style={{
            boxShadow: activeTab === tab.id ? 'var(--shadow-soft-outer)' : 'none'
          }}
        >
          {tab.icon}
          <span>{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
