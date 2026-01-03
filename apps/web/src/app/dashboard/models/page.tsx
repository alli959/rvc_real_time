'use client';

import { InProgress } from '@/components/in-progress';
import { DashboardLayout } from '@/components/dashboard-layout';

export default function MyModelsPage() {
  return (
    <DashboardLayout>
      <InProgress 
        title="My Models"
        description="Your personal voice models collection is coming soon. You'll be able to upload, train, and manage your own custom voice models."
      />
    </DashboardLayout>
  );
}
