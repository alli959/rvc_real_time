'use client';

import { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { roleRequestsApi, RoleInfo, RoleRequest } from '@/lib/api';
import { useAuthStore } from '@/lib/store';
import {
  Settings,
  User,
  Shield,
  Key,
  Bell,
  Send,
  Check,
  X,
  Clock,
  AlertCircle,
  ChevronRight,
  Upload,
  Wand2,
  Globe,
} from 'lucide-react';

type SettingsTab = 'profile' | 'roles' | 'security' | 'notifications';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('profile');
  const { user, hasRole } = useAuthStore();

  const tabs = [
    { id: 'profile' as SettingsTab, label: 'Profile', icon: User },
    { id: 'roles' as SettingsTab, label: 'Roles & Permissions', icon: Shield },
    { id: 'security' as SettingsTab, label: 'Security', icon: Key },
    { id: 'notifications' as SettingsTab, label: 'Notifications', icon: Bell },
  ];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Settings className="h-7 w-7" />
            Settings
          </h1>
          <p className="text-gray-400 mt-1">Manage your account settings and preferences</p>
        </div>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar */}
          <nav className="lg:w-56 flex lg:flex-col gap-1 overflow-x-auto lg:overflow-visible">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-3 px-4 py-2 rounded-lg transition-colors whitespace-nowrap ${
                  activeTab === id
                    ? 'bg-primary-600/20 text-primary-400'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                }`}
              >
                <Icon className="h-5 w-5" />
                {label}
              </button>
            ))}
          </nav>

          {/* Content */}
          <div className="flex-1">
            {activeTab === 'profile' && <ProfileSettings />}
            {activeTab === 'roles' && <RolesSettings />}
            {activeTab === 'security' && <SecuritySettings />}
            {activeTab === 'notifications' && <NotificationSettings />}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

function ProfileSettings() {
  const { user } = useAuthStore();

  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Profile Information</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Name</label>
            <input
              type="text"
              defaultValue={user?.name}
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Email</label>
            <input
              type="email"
              defaultValue={user?.email}
              disabled
              className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-gray-400 cursor-not-allowed"
            />
            <p className="text-xs text-gray-500 mt-1">Contact support to change your email</p>
          </div>

          <div className="pt-4">
            <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function RolesSettings() {
  const { user, hasRole } = useAuthStore();
  const [availableRoles, setAvailableRoles] = useState<Record<string, RoleInfo>>({});
  const [myRequests, setMyRequests] = useState<RoleRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [requestRole, setRequestRole] = useState<string | null>(null);
  const [requestMessage, setRequestMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [rolesData, requestsData] = await Promise.all([
        roleRequestsApi.getAvailableRoles(),
        roleRequestsApi.myRequests(),
      ]);
      setAvailableRoles(rolesData.roles);
      setMyRequests(requestsData.requests);
    } catch (err) {
      console.error('Failed to load role data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitRequest = async () => {
    if (!requestRole || !requestMessage.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      await roleRequestsApi.create({ role: requestRole, message: requestMessage });
      await loadData();
      setRequestRole(null);
      setRequestMessage('');
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to submit request');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancelRequest = async (id: number) => {
    try {
      await roleRequestsApi.cancel(id);
      await loadData();
    } catch (err) {
      console.error('Failed to cancel request:', err);
    }
  };

  const getRoleIcon = (role: string) => {
    switch (role) {
      case 'creator':
        return <Upload className="h-5 w-5" />;
      case 'trainer':
        return <Wand2 className="h-5 w-5" />;
      case 'publisher':
        return <Globe className="h-5 w-5" />;
      default:
        return <Shield className="h-5 w-5" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return (
          <span className="flex items-center gap-1 text-xs px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded">
            <Clock className="h-3 w-3" />
            Pending
          </span>
        );
      case 'approved':
        return (
          <span className="flex items-center gap-1 text-xs px-2 py-1 bg-green-500/20 text-green-400 rounded">
            <Check className="h-3 w-3" />
            Approved
          </span>
        );
      case 'rejected':
        return (
          <span className="flex items-center gap-1 text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded">
            <X className="h-3 w-3" />
            Rejected
          </span>
        );
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Current Roles */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Your Current Roles</h2>
        <div className="flex flex-wrap gap-2">
          {user?.roles?.map((role) => (
            <span
              key={role}
              className="flex items-center gap-2 px-3 py-1.5 bg-primary-600/20 text-primary-400 rounded-full text-sm"
            >
              <Shield className="h-4 w-4" />
              {role.charAt(0).toUpperCase() + role.slice(1)}
            </span>
          )) || (
            <span className="text-gray-500">No special roles assigned</span>
          )}
        </div>
      </div>

      {/* Available Roles */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Request Additional Roles</h2>
        <p className="text-gray-400 text-sm mb-6">
          Request access to additional features by applying for these roles. An administrator will review your request.
        </p>

        <div className="space-y-4">
          {Object.entries(availableRoles).map(([roleKey, roleInfo]) => {
            const hasPendingRequest = myRequests.some(
              (r) => r.requested_role === roleKey && r.status === 'pending'
            );
            
            return (
              <div
                key={roleKey}
                className={`border rounded-lg p-4 transition-colors ${
                  roleInfo.has_role
                    ? 'border-green-500/30 bg-green-500/5'
                    : 'border-gray-700 hover:border-gray-600'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg ${
                      roleInfo.has_role ? 'bg-green-500/20 text-green-400' : 'bg-gray-800 text-gray-400'
                    }`}>
                      {getRoleIcon(roleKey)}
                    </div>
                    <div>
                      <h3 className="font-medium text-white flex items-center gap-2">
                        {roleInfo.name}
                        {roleInfo.has_role && (
                          <Check className="h-4 w-4 text-green-400" />
                        )}
                      </h3>
                      <p className="text-sm text-gray-400 mt-1">{roleInfo.description}</p>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {roleInfo.permissions.map((perm) => (
                          <span key={perm} className="text-xs px-2 py-0.5 bg-gray-800 text-gray-500 rounded">
                            {perm.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    {roleInfo.has_role ? (
                      <span className="text-xs text-green-400">Active</span>
                    ) : hasPendingRequest ? (
                      <span className="text-xs text-yellow-400">Pending</span>
                    ) : (
                      <button
                        onClick={() => setRequestRole(roleKey)}
                        className="text-sm px-3 py-1 bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
                      >
                        Request
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Request Modal */}
      {requestRole && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <h2 className="text-lg font-semibold text-white mb-4">
              Request {availableRoles[requestRole]?.name} Role
            </h2>
            
            {error && (
              <div className="flex items-center gap-2 text-sm text-red-400 bg-red-500/10 px-3 py-2 rounded-lg mb-4">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            )}
            
            <p className="text-sm text-gray-400 mb-4">
              Please explain why you need this role and how you plan to use it.
            </p>
            
            <textarea
              value={requestMessage}
              onChange={(e) => setRequestMessage(e.target.value)}
              placeholder="Enter your request message..."
              className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none"
              rows={4}
            />

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  setRequestRole(null);
                  setRequestMessage('');
                  setError(null);
                }}
                className="flex-1 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitRequest}
                disabled={submitting || !requestMessage.trim()}
                className="flex-1 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                {submitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                    Submitting...
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    Submit Request
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* My Requests */}
      {myRequests.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Your Role Requests</h2>
          
          <div className="space-y-3">
            {myRequests.map((request) => (
              <div key={request.id} className="border border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    {getRoleIcon(request.requested_role)}
                    <span className="font-medium text-white">
                      {availableRoles[request.requested_role]?.name || request.requested_role}
                    </span>
                    {getStatusBadge(request.status)}
                  </div>
                  {request.status === 'pending' && (
                    <button
                      onClick={() => handleCancelRequest(request.id)}
                      className="text-sm text-gray-400 hover:text-red-400 transition-colors"
                    >
                      Cancel
                    </button>
                  )}
                </div>
                
                <p className="text-sm text-gray-400 mb-2">{request.message}</p>
                
                {request.admin_response && (
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    <p className="text-xs text-gray-500 mb-1">Admin Response:</p>
                    <p className="text-sm text-gray-300">{request.admin_response}</p>
                  </div>
                )}
                
                <p className="text-xs text-gray-500 mt-2">
                  Requested {new Date(request.created_at).toLocaleDateString()}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SecuritySettings() {
  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Change Password</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Current Password</label>
            <input
              type="password"
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">New Password</label>
            <input
              type="password"
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Confirm New Password</label>
            <input
              type="password"
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div className="pt-4">
            <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
              Update Password
            </button>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-white mb-4">API Tokens</h2>
        <p className="text-gray-400 text-sm mb-4">
          Manage API tokens for programmatic access to your account.
        </p>
        <button className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors">
          Generate New Token
        </button>
      </div>
    </div>
  );
}

function NotificationSettings() {
  return (
    <div className="space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Email Notifications</h2>
        
        <div className="space-y-4">
          <label className="flex items-center justify-between">
            <div>
              <span className="text-white">Job Completion</span>
              <p className="text-sm text-gray-500">Get notified when your voice conversions complete</p>
            </div>
            <input type="checkbox" defaultChecked className="rounded border-gray-700 bg-gray-800 text-primary-600" />
          </label>

          <label className="flex items-center justify-between">
            <div>
              <span className="text-white">Role Request Updates</span>
              <p className="text-sm text-gray-500">Get notified when your role requests are reviewed</p>
            </div>
            <input type="checkbox" defaultChecked className="rounded border-gray-700 bg-gray-800 text-primary-600" />
          </label>

          <label className="flex items-center justify-between">
            <div>
              <span className="text-white">Product Updates</span>
              <p className="text-sm text-gray-500">Stay informed about new features and improvements</p>
            </div>
            <input type="checkbox" className="rounded border-gray-700 bg-gray-800 text-primary-600" />
          </label>
        </div>

        <div className="pt-4">
          <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors">
            Save Preferences
          </button>
        </div>
      </div>
    </div>
  );
}
