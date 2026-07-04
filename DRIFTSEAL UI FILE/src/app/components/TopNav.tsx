import React from 'react';
import { useNavigate, useLocation } from 'react-router';
import { ChevronLeft, User, Home } from 'lucide-react';
import { IconContainer } from './skeuomorphic/IconContainer';
import { InsetPanel } from './skeuomorphic/InsetPanel';
import { motion, AnimatePresence } from 'motion/react';

interface TopNavProps {
  pageTitle: string;
}

export function TopNav({ pageTitle }: TopNavProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [showUserMenu, setShowUserMenu] = React.useState(false);

  const getBackRoute = () => {
    const path = location.pathname;
    
    // Dashboard → Upload
    if (path === '/dashboard') return '/upload';
    
    // Upload → Landing
    if (path === '/upload') return '/';
    
    // Forensic / Testing / Remediation / History / Monitoring / Settings → Dashboard
    if (['/forensic', '/testing', '/remediation', '/history', '/monitoring', '/settings'].includes(path)) {
      return '/dashboard';
    }
    
    // Default: try browser history or go to landing
    return '/';
  };

  const handleBack = () => {
    const backRoute = getBackRoute();
    navigate(backRoute);
  };

  return (
    <div className="mb-6">
      <InsetPanel size="sm">
        <div className="flex items-center justify-between gap-4">
          {/* Left: Back Button + Title */}
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <button
              onClick={handleBack}
              className="flex-shrink-0 group"
              aria-label="Go back"
            >
              <IconContainer 
                size="md" 
                variant="raised"
                className="transition-all group-hover:shadow-[var(--shadow-soft-outer-lg)] group-active:shadow-[var(--shadow-soft-inset)]"
              >
                <ChevronLeft className="w-5 h-5 text-[#A0D8F1] group-hover:text-[#7BC5E8]" />
              </IconContainer>
            </button>
            
            <h1 className="text-[#2C3E50] truncate">{pageTitle}</h1>
          </div>

          {/* Right: User Menu */}
          <div className="flex items-center gap-3 flex-shrink-0 relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="group"
              aria-label="User menu"
            >
              <IconContainer 
                size="md" 
                variant="raised"
                color="#CDB4DB"
                className="transition-all group-hover:shadow-[var(--shadow-soft-outer-lg)]"
              >
                <User className="w-5 h-5 text-white" />
              </IconContainer>
            </button>

            {/* User Dropdown */}
            <AnimatePresence>
              {showUserMenu && (
                <>
                  {/* Backdrop */}
                  <div 
                    className="fixed inset-0 z-40"
                    onClick={() => setShowUserMenu(false)}
                  />
                  
                  {/* Dropdown Menu */}
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                    className="absolute right-0 top-full mt-2 z-50 min-w-[200px]"
                  >
                    <InsetPanel className="bg-white">
                      <div className="space-y-2">
                        {/* User Info */}
                        <div className="px-3 py-2 border-b border-[#E8EDF2]">
                          <p className="text-sm font-medium text-[#2C3E50]">ML Security Team</p>
                          <p className="text-xs text-[#6B7C8F]">security@company.com</p>
                        </div>

                        {/* Menu Items */}
                        <div className="space-y-1">
                          <button
                            onClick={() => {
                              setShowUserMenu(false);
                              navigate('/settings');
                            }}
                            className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-[#6B7C8F] hover:text-[#2C3E50] hover:bg-[#F0F4F8] transition-all text-left"
                          >
                            <User className="w-4 h-4" />
                            <span className="text-sm">Profile Settings</span>
                          </button>

                          <button
                            onClick={() => {
                              setShowUserMenu(false);
                              navigate('/');
                            }}
                            className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-[#6B7C8F] hover:text-[#2C3E50] hover:bg-[#F0F4F8] transition-all text-left"
                          >
                            <Home className="w-4 h-4" />
                            <span className="text-sm">Return to Landing</span>
                          </button>
                        </div>
                      </div>
                    </InsetPanel>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
        </div>
      </InsetPanel>
    </div>
  );
}
