'use client';

import { InProgress } from '@/components/in-progress';
import { DashboardLayout } from '@/components/dashboard-layout';

export default function SettingsPage() {
  return (
    <DashboardLayout>
      <InProgress 
        title="Settings"
        description="User settings and preferences are coming soon. You'll be able to customize your experience, manage API keys, and configure default conversion parameters."
      />
    </DashboardLayout>
  );
}
