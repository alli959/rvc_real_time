'use client';

import { Construction, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

interface InProgressProps {
  title: string;
  description?: string;
  backHref?: string;
  backLabel?: string;
}

export function InProgress({ 
  title, 
  description = "This feature is currently under development.", 
  backHref = "/dashboard",
  backLabel = "Back to Dashboard"
}: InProgressProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4">
      <div className="glass rounded-2xl p-8 max-w-md w-full text-center">
        {/* Icon */}
        <div className="w-16 h-16 rounded-full bg-yellow-500/10 flex items-center justify-center mx-auto mb-6">
          <Construction className="h-8 w-8 text-yellow-500" />
        </div>

        {/* Title */}
        <h1 className="text-2xl font-bold mb-2">{title}</h1>

        {/* Description */}
        <p className="text-gray-400 mb-6">
          {description}
        </p>

        {/* Progress Indicator */}
        <div className="flex items-center justify-center gap-2 mb-6">
          <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
          <span className="text-sm text-yellow-500 font-medium">In Progress</span>
        </div>

        {/* Back Button */}
        <Link
          href={backHref}
          className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          {backLabel}
        </Link>
      </div>
    </div>
  );
}

export default InProgress;
