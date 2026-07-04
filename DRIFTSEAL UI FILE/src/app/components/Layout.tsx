import React from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router';
import { Shield, LayoutDashboard, Search, FlaskConical, Sparkles, History, Activity, Settings, Menu, Upload, Home } from 'lucide-react';
import { InsetPanel } from './skeuomorphic/InsetPanel';

export function Layout() {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = React.useState(true);

  const navItems = [
    { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/upload', icon: Upload, label: 'Upload' },
    { to: '/forensic', icon: Search, label: 'Forensic Analysis' },
    { to: '/testing', icon: FlaskConical, label: 'Testing & Validation' },
    { to: '/remediation', icon: Sparkles, label: 'Remediation' },
    { to: '/monitoring', icon: Activity, label: 'Monitoring' },
    { to: '/history', icon: History, label: 'Scan History' },
    { to: '/settings', icon: Settings, label: 'Settings' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#F7F9FB] via-[#F0F4F8] to-[#E8EDF2] flex flex-col lg:flex-row">
      {/* Sidebar */}
      <aside 
        className={`${sidebarOpen ? 'lg:w-64' : 'lg:w-20'} w-full lg:flex-shrink-0 transition-all duration-300 p-4 lg:p-6`}
      >
        <InsetPanel className="h-full flex flex-col">
          {/* Logo */}
          <button 
            onClick={() => navigate('/')}
            className="flex items-center gap-3 mb-6 lg:mb-8 px-2 hover:opacity-80 transition-opacity"
          >
            <div className="w-8 h-8 flex items-center justify-center flex-shrink-0">
              <Shield className="w-8 h-8 text-[#A0D8F1]" />
            </div>
            {sidebarOpen && (
              <div className="hidden lg:block text-left">
                <h2 className="text-[#2C3E50]">Drift Seal</h2>
                <p className="text-xs text-[#6B7C8F]">v2.0</p>
              </div>
            )}
          </button>

          {/* Navigation */}
          <nav className="flex-1 space-y-2 overflow-y-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center rounded-2xl transition-all ${
                      sidebarOpen 
                        ? 'gap-3 px-4 py-3' 
                        : 'justify-center px-3 py-3 lg:px-4'
                    } ${
                      isActive
                        ? 'bg-[#A0D8F1] text-[#1A4D6E]'
                        : 'text-[#6B7C8F] hover:text-[#2C3E50] hover:bg-white/50'
                    }`
                  }
                  style={({ isActive }) => ({
                    boxShadow: isActive ? 'var(--shadow-soft-outer)' : 'none'
                  })}
                  title={!sidebarOpen ? item.label : undefined}
                >
                  <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                    <Icon className="w-5 h-5" />
                  </div>
                  {sidebarOpen && <span className="truncate">{item.label}</span>}
                </NavLink>
              );
            })}
          </nav>

          {/* Bottom Actions */}
          <div className="space-y-2 mt-4 pt-4 border-t border-[#E8EDF2]">
            {/* Return to Landing */}
            <button
              onClick={() => navigate('/')}
              className={`w-full flex items-center rounded-2xl text-[#6B7C8F] hover:text-[#2C3E50] hover:bg-white/50 transition-all ${
                sidebarOpen ? 'gap-3 px-4 py-3' : 'justify-center px-3 py-3'
              }`}
              title={!sidebarOpen ? 'Return to Landing' : undefined}
            >
              <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                <Home className="w-5 h-5" />
              </div>
              {sidebarOpen && <span className="truncate">Return to Landing</span>}
            </button>

            {/* Toggle Button */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`hidden lg:flex w-full rounded-2xl text-[#6B7C8F] hover:text-[#2C3E50] hover:bg-white/50 transition-all items-center ${
                sidebarOpen ? 'gap-3 px-4 py-3' : 'justify-center px-3 py-3'
              }`}
              title={!sidebarOpen ? 'Expand' : 'Collapse'}
            >
              <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                <Menu className="w-5 h-5" />
              </div>
              {sidebarOpen && <span>Collapse</span>}
            </button>
          </div>
        </InsetPanel>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-4 lg:p-6 overflow-auto">
        <div className="max-w-7xl mx-auto w-full">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
