import { createBrowserRouter, Navigate } from 'react-router';
import { LandingPage } from './pages/LandingPage';
import { UploadPage } from './pages/UploadPage';
import { DashboardPage } from './pages/DashboardPage';
import { ForensicPage } from './pages/ForensicPage';
import { TestingPage } from './pages/TestingPage';
import { RemediationPage } from './pages/RemediationPage';
import { ScanHistoryPage } from './pages/ScanHistoryPage';
import { MonitoringPage } from './pages/MonitoringPage';
import { SettingsPage } from './pages/SettingsPage';
import { Layout } from './components/Layout';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <LandingPage />
  },
  {
    path: '/login',
    element: <Navigate to="/" replace />
  },
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        path: 'upload',
        element: <UploadPage />
      },
      {
        path: 'dashboard',
        element: <DashboardPage />
      },
      {
        path: 'forensic',
        element: <ForensicPage />
      },
      {
        path: 'testing',
        element: <TestingPage />
      },
      {
        path: 'remediation',
        element: <RemediationPage />
      },
      {
        path: 'history',
        element: <ScanHistoryPage />
      },
      {
        path: 'monitoring',
        element: <MonitoringPage />
      },
      {
        path: 'settings',
        element: <SettingsPage />
      }
    ]
  }
]);
