'use client';

import { InProgress } from '@/components/in-progress';
import { DashboardLayout } from '@/components/dashboard-layout';

export default function JobsPage() {
  return (
    <DashboardLayout>
      <InProgress 
        title="My Jobs"
        description="Job history and queue management is coming soon. You'll be able to view past conversions, track ongoing jobs, and manage your queue."
      />
    </DashboardLayout>
  );
}
