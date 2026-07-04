import React, { useState } from 'react';
import { Settings, Shield, Bell, Lock, Database, Eye, Save, Key } from 'lucide-react';
import { TopNav } from '../components/TopNav';
import { RaisedCard } from '../components/skeuomorphic/RaisedCard';
import { RaisedButton } from '../components/skeuomorphic/RaisedButton';
import { SoftToggle } from '../components/skeuomorphic/SoftToggle';
import { SoftSlider } from '../components/skeuomorphic/SoftSlider';
import { InsetInput } from '../components/skeuomorphic/InsetInput';
import { InsetPanel } from '../components/skeuomorphic/InsetPanel';

export function SettingsPage() {
  const [accuracyThreshold, setAccuracyThreshold] = useState(85);
  const [f1Threshold, setF1Threshold] = useState(80);
  const [emailAlerts, setEmailAlerts] = useState(true);
  const [slackAlerts, setSlackAlerts] = useState(false);

  return (
    <div className="space-y-6">
      <TopNav pageTitle="Settings" />

      {/* Header */}
      <div>
        <h1 className="mb-2 text-[#2C3E50]">Settings</h1>
        <p className="text-[#6B7C8F]">Configure detection thresholds and system preferences</p>
      </div>

      {/* Detection Thresholds */}
      <RaisedCard>
        <div className="flex items-center gap-3 mb-6">
          <Shield className="w-6 h-6 text-[#A0D8F1]" />
          <h2 className="text-[#2C3E50]">Detection Thresholds</h2>
        </div>

        <InsetPanel>
          <div className="space-y-6">
            <SoftSlider
              value={accuracyThreshold}
              onChange={setAccuracyThreshold}
              min={50}
              max={100}
              label="Minimum Accuracy Threshold"
              unit="%"
            />

            <SoftSlider
              value={f1Threshold}
              onChange={setF1Threshold}
              min={50}
              max={100}
              label="Minimum F1-Score Threshold"
              unit="%"
            />
          </div>
        </InsetPanel>

        <div className="mt-6 flex justify-end">
          <RaisedButton variant="primary">
            Save Thresholds
          </RaisedButton>
        </div>
      </RaisedCard>

      {/* Notification Settings */}
      <RaisedCard>
        <div className="flex items-center gap-3 mb-6">
          <Bell className="w-6 h-6 text-[#FFD3B6]" />
          <h2 className="text-[#2C3E50]">Notifications</h2>
        </div>

        <InsetPanel>
          <div className="space-y-4">
            <SoftToggle
              checked={emailAlerts}
              onCheckedChange={setEmailAlerts}
              label="Email Alerts"
            />
            
            <SoftToggle
              checked={slackAlerts}
              onCheckedChange={setSlackAlerts}
              label="Slack Notifications"
            />

            {slackAlerts && (
              <div className="pt-4 border-t border-[#E8EDF2]">
                <InsetInput
                  placeholder="https://hooks.slack.com/services/..."
                  label="Slack Webhook URL"
                />
              </div>
            )}
          </div>
        </InsetPanel>
      </RaisedCard>

      {/* API Keys */}
      <RaisedCard>
        <div className="flex items-center gap-3 mb-6">
          <Key className="w-6 h-6 text-[#A8E6CF]" />
          <h2 className="text-[#2C3E50]">API Access</h2>
        </div>

        <InsetPanel>
          <div className="space-y-4">
            <InsetInput
              type="password"
              placeholder="••••••••••••••••"
              label="API Key"
              value="sk_live_1234567890abcdef"
              readOnly
            />

            <div className="flex gap-3">
              <RaisedButton variant="outline" size="sm">
                Regenerate
              </RaisedButton>
              <RaisedButton variant="outline" size="sm">
                Copy
              </RaisedButton>
            </div>

            <div className="pt-4 border-t border-[#E8EDF2] text-xs text-[#6B7C8F]">
              <p>⚠️ Keep your API key secure. Do not share it publicly.</p>
            </div>
          </div>
        </InsetPanel>
      </RaisedCard>
    </div>
  );
}