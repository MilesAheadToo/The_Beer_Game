import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Line, Pie } from 'react-chartjs-2';
import {
  UsersIcon,
  ClockIcon,
  ChartBarIcon,
  UserGroupIcon,
  CogIcon,
  ShieldCheckIcon,
  DocumentTextIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';
import { mixedGameApi } from '../../services/api';

ChartJS.register(CategoryScale, LinearScale, LineElement, PointElement, Title, Tooltip, Legend, ArcElement);

const StatCard = ({ title, value, icon: Icon, change, changeType = 'neutral', loading = false }) => (
  <div className="card-surface overflow-hidden rounded-lg">
    <div className="pad-6">
      <div className="flex items-center">
        <div className="flex-shrink-0 bg-indigo-500 rounded-md p-3">
          <Icon className="h-6 w-6 text-white" aria-hidden="true" />
        </div>
        <div className="ml-5 w-0 flex-1">
          <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
          {loading ? (
            <dd className="animate-pulse h-7 bg-gray-200 rounded w-3/4"></dd>
          ) : (
            <dd className="flex items-baseline">
              <div className="text-2xl font-semibold text-gray-900">{value}</div>
              {typeof change === 'number' && (
                <div
                  className={`ml-2 flex items-baseline text-sm font-medium ${
                    changeType === 'increase' ? 'text-green-600' : changeType === 'decrease' ? 'text-red-600' : 'text-gray-500'
                  }`}
                >
                  {changeType === 'increase' ? <span className="text-green-500">↑</span> : changeType === 'decrease' ? <span className="text-red-500">↓</span> : null}
                  <span className="ml-1">{change}%</span>
                </div>
              )}
            </dd>
          )}
        </div>
      </div>
    </div>
  </div>
);

const UserStatusBadge = ({ status }) => {
  const statusMap = {
    active: { text: 'Active', color: 'bg-green-100 text-green-800' },
    inactive: { text: 'Inactive', color: 'bg-yellow-100 text-yellow-800' },
    suspended: { text: 'Suspended', color: 'bg-red-100 text-red-800' },
    pending: { text: 'Pending', color: 'bg-blue-100 text-blue-800' },
  };
  const key = typeof status === 'string' ? status.toLowerCase() : 'pending';
  const statusInfo = statusMap[key] || { text: String(status ?? 'Unknown'), color: 'bg-gray-100 text-gray-800' };
  return <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.color}`}>{statusInfo.text}</span>;
};

const GameStatusBadge = ({ status }) => {
  const statusMap = {
    completed: { text: 'Completed', color: 'bg-green-100 text-green-800' },
    in_progress: { text: 'In Progress', color: 'bg-blue-100 text-blue-800' },
    abandoned: { text: 'Abandoned', color: 'bg-red-100 text-red-800' },
    waiting: { text: 'Waiting', color: 'bg-yellow-100 text-yellow-800' },
  };
  const statusInfo = statusMap[status] || { text: status, color: 'bg-gray-100 text-gray-800' };
  return <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.color}`}>{statusInfo.text}</span>;
};

const RangesModal = ({ visible, rangeEdits, onClose, onChange, onSave, originalConfig }) => {
  if (!visible) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <h4 className="text-base font-semibold">Edit System Ranges</h4>
          <button className="text-gray-500 hover:text-gray-700" onClick={onClose}>✕</button>
        </div>
        <div className="p-4 max-h-[70vh] overflow-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(rangeEdits || {}).map(([k, rng]) => {
              const orig = (originalConfig || {})[k] || {};
              const changed = rng?.min !== orig?.min || rng?.max !== orig?.max;
              const invalid = Number.isFinite(rng?.min) && Number.isFinite(rng?.max) && rng.min > rng.max;
              return (
                <div key={k} className={`rounded border ${invalid ? 'border-red-300 bg-red-50' : changed ? 'border-yellow-300 bg-yellow-50' : 'border-gray-200 bg-gray-50'} p-3`}>
                  <div className="text-sm text-gray-700 mb-2 capitalize">{k.replaceAll('_', ' ')}</div>
                  <div className="flex items-center space-x-2">
                    <div className="flex-1">
                      <label className="block text-xs text-gray-500 mb-1">Min</label>
                      <input type="number" className="w-full border border-gray-300 rounded px-2 py-1 text-sm" value={rng.min} onChange={(e) => onChange(k, { ...rng, min: e.target.valueAsNumber })} />
                    </div>
                    <div className="flex-1">
                      <label className="block text-xs text-gray-500 mb-1">Max</label>
                      <input type="number" className="w-full border border-gray-300 rounded px-2 py-1 text-sm" value={rng.max} onChange={(e) => onChange(k, { ...rng, max: e.target.valueAsNumber })} />
                    </div>
                  </div>
                  {invalid && <div className="mt-1 text-xs text-red-600">Min must be ≤ Max</div>}
                </div>
              );
            })}
          </div>
        </div>
        <div className="px-4 py-3 border-t border-gray-200 flex justify-end space-x-2">
          <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={onClose}>Cancel</button>
          <button className="text-white bg-indigo-600 px-3 py-1 rounded text-sm disabled:opacity-50" disabled={Object.values(rangeEdits || {}).some((r) => (r?.min ?? 0) > (r?.max ?? 0))} onClick={onSave}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

const AdminDashboard = () => {
  const { isAdmin } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  const [stats, setStats] = useState(null);
  const [recentUsers, setRecentUsers] = useState([]);
  const [recentGames, setRecentGames] = useState([]);
  const [gameConfigs, setGameConfigs] = useState({});

  const [showRangesModal, setShowRangesModal] = useState(false);
  const [rangeEdits, setRangeEdits] = useState({});

  useEffect(() => {
    if (!isAdmin) navigate('/unauthorized');
  }, [isAdmin, navigate]);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        setIsLoading(true);
        const mockStats = {
          totalUsers: 1245,
          activeUsers: 342,
          totalGames: 876,
          activeGames: 23,
          avgGameDuration: 45,
          userGrowth: 12.5,
          gameGrowth: 8.2,
          userActivity: { labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], data: [65, 59, 80, 81, 56, 55, 40] },
          gameStats: { completed: 65, inProgress: 23, abandoned: 12 },
          userDistribution: { new: 45, returning: 30, inactive: 25 },
        };
        const mockRecentUsers = [
          { id: 1, username: 'johndoe', email: 'john@example.com', joined: '2023-05-15T10:30:00Z', status: 'active' },
          { id: 2, username: 'alice_smith', email: 'alice@example.com', joined: '2023-05-14T15:45:00Z', status: 'active' },
          { id: 3, username: 'bob_johnson', email: 'bob@example.com', joined: '2023-05-13T09:20:00Z', status: 'inactive' },
        ];
        setStats(mockStats);
        setRecentUsers(mockRecentUsers);

        try {
          const games = await mixedGameApi.getGames();
          setRecentGames(games || []);
          const map = {};
          (games || []).forEach((g) => {
            map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {} };
          });
          setGameConfigs(map);
        } catch {
          setRecentGames([]);
        }
        setError(null);
      } catch (err) {
        setError('Failed to load admin dashboard. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    fetchAll();
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get('openSystemRanges') === '1') {
      (async () => {
        try {
          const cfg = await mixedGameApi.getSystemConfig();
          setRangeEdits(cfg || {});
        } catch {
          setRangeEdits({});
        } finally {
          setShowRangesModal(true);
        }
      })();
    }
  }, [location.search]);

  const userActivityData = {
    labels: stats?.userActivity?.labels || [],
    datasets: [
      {
        label: 'Active Users',
        data: stats?.userActivity?.data || [],
        borderColor: 'rgb(99, 102, 241)',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        tension: 0.3,
        fill: true,
      },
    ],
  };

  const gameStatusData = {
    labels: ['Completed', 'In Progress', 'Abandoned'],
    datasets: [
      {
        data: [stats?.gameStats?.completed || 0, stats?.gameStats?.inProgress || 0, stats?.gameStats?.abandoned || 0],
        backgroundColor: ['rgba(16, 185, 129, 0.8)', 'rgba(59, 130, 246, 0.8)', 'rgba(239, 68, 68, 0.8)'],
        borderColor: ['rgba(16, 185, 129, 1)', 'rgba(59, 130, 246, 1)', 'rgba(239, 68, 68, 1)'],
        borderWidth: 1,
      },
    ],
  };

  const userDistributionData = {
    labels: ['New Users', 'Returning Users', 'Inactive Users'],
    datasets: [
      {
        data: [stats?.userDistribution?.new || 0, stats?.userDistribution?.returning || 0, stats?.userDistribution?.inactive || 0],
        backgroundColor: ['rgba(99, 102, 241, 0.8)', 'rgba(139, 92, 246, 0.8)', 'rgba(156, 163, 175, 0.8)'],
        borderColor: ['rgba(99, 102, 241, 1)', 'rgba(139, 92, 246, 1)', 'rgba(156, 163, 175, 1)'],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = { responsive: true, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } };

  const formatDateTime = (iso) => {
    if (!iso) return '-';
    try {
      const d = new Date(iso);
      return `${d.toLocaleDateString()} ${d.toLocaleTimeString()}`;
    } catch {
      return '-';
    }
  };
  const humanDuration = (startIso, endIso) => {
    try {
      const start = startIso ? new Date(startIso).getTime() : null;
      const end = endIso ? new Date(endIso).getTime() : Date.now();
      if (!start) return '-';
      const mins = Math.floor(Math.max(0, end - start) / 60000);
      const hrs = Math.floor(mins / 60);
      const rem = mins % 60;
      return hrs > 0 ? `${hrs}h ${rem}m` : `${mins}m`;
    } catch {
      return '-';
    }
  };

  const { user } = useAuth();
  // Games created by this admin for selection
  const [myGames, setMyGames] = useState([]);
  const location = useLocation();
  const navigate = useNavigate();
  const params = new URLSearchParams(location.search);
  const selectedGameId = params.get('gameId') || '';

  useEffect(() => {
    (async () => {
      try {
        const games = await mixedGameApi.getGames();
        const mine = (games || []).filter(g => g.created_by === user?.id);
        setMyGames(mine);
      } catch (e) {
        // ignore
      }
    })();
  }, [user?.id]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="card-surface pad-6 rounded-lg h-32"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="bg-red-50 border-l-4 border-red-400 p-4">{error}</div>
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="card-surface">
          <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center">
              <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
              <div className="flex space-x-3">
                {/* Game selector (only games created by this admin) */}
                {myGames.length > 0 && (
                  <select
                    className="block pl-3 pr-10 py-2 text-sm border-gray-300 rounded-md"
                    value={selectedGameId}
                    onChange={(e) => navigate(`/admin?gameId=${e.target.value}`)}
                    title="Select a game to view context-aware dashboards"
                  >
                    <option value="">Select Game…</option>
                    {myGames.map((g) => (
                      <option key={g.id} value={g.id}>{g.name}</option>
                    ))}
                  </select>
                )}
                <button type="button" onClick={() => navigate('/users')} className="inline-flex items-center px-4 py-2 text-sm rounded-md text-white bg-indigo-600">
                  <UsersIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                  Users
                </button>
                <button type="button" onClick={() => setShowRangesModal(true)} className="inline-flex items-center px-4 py-2 text-sm rounded-md text-white bg-blue-600">
                  <CogIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                  Edit Ranges
                </button>
                <button type="button" onClick={() => navigate('/admin/model-setup')} className="inline-flex items-center px-4 py-2 text-sm rounded-md text-white bg-indigo-600">
                  <CogIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                  Model Setup
                </button>
                <button type="button" onClick={() => navigate('/admin/training')} className="inline-flex items-center px-4 py-2 text-sm rounded-md text-white bg-green-600">
                  Training
                </button>
              </div>
            </div>
            {/* Tabs */}
            <div className="mt-6 border-b border-gray-200">
              <nav className="-mb-px flex space-x-8">
                {[
                  { name: 'Overview', id: 'overview', icon: ChartBarIcon },
                  { name: 'Users', id: 'users', icon: UsersIcon },
                  { name: 'Games', id: 'games', icon: UserGroupIcon },
                  { name: 'Security', id: 'security', icon: ShieldCheckIcon },
                  { name: 'Reports', id: 'reports', icon: DocumentTextIcon },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`${activeTab === tab.id ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}
                  >
                    <tab.icon className="mr-2 h-5 w-5" />
                    {tab.name}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Overview */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
                <StatCard title="Total Users" value={stats?.totalUsers?.toLocaleString() || '0'} icon={UsersIcon} change={stats?.userGrowth} changeType={stats?.userGrowth >= 0 ? 'increase' : 'decrease'} />
                <StatCard title="Active Users" value={stats?.activeUsers?.toLocaleString() || '0'} icon={UserCircleIcon} />
                <StatCard title="Total Games" value={stats?.totalGames?.toLocaleString() || '0'} icon={UserGroupIcon} change={stats?.gameGrowth} changeType={stats?.gameGrowth >= 0 ? 'increase' : 'decrease'} />
                <StatCard title="Active Games" value={stats?.activeGames?.toLocaleString() || '0'} icon={ClockIcon} />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card-surface pad-6 rounded-lg">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">User Activity (Last 7 Days)</h3>
                  <div className="h-80">
                    <Line data={userActivityData} options={chartOptions} />
                  </div>
                </div>
                <div className="grid grid-rows-2 gap-6">
                  <div className="card-surface pad-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Game Status</h3>
                    <div className="h-64">
                      <Pie data={gameStatusData} options={chartOptions} />
                    </div>
                  </div>
                  <div className="card-surface pad-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">User Distribution</h3>
                    <div className="h-64">
                      <Pie data={userDistributionData} options={chartOptions} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Users */}
          {activeTab === 'users' && (
            <div className="card-surface overflow-hidden sm:rounded-lg">
              <div className="pad-6 border-b border-gray-200">
                <h3 className="text-lg leading-6 font-medium text-gray-900">User Management</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Joined</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {recentUsers.map((user) => (
                      <tr key={user.id}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-shrink-0 h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center">
                              <UserCircleIcon className="h-8 w-8 text-indigo-500" />
                            </div>
                            <div className="ml-4">
                              <div className="text-sm font-medium text-gray-900">{user.username}</div>
                              <div className="text-sm text-gray-500">ID: {user.id}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">{user.email}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">{new Date(user.joined).toLocaleDateString()}</div>
                          <div className="text-sm text-gray-500">{new Date(user.joined).toLocaleTimeString()}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <UserStatusBadge status={user.status} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Games */}
          {activeTab === 'games' && (
            <div className="table-surface overflow-hidden sm:rounded-lg">
              <div className="pad-6 border-b border-gray-200">
                <h3 className="text-lg leading-6 font-medium text-gray-900">Game Management</h3>
                <p className="mt-1 max-w-2xl text-sm text-gray-500">View and manage active and completed games</p>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Game</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Players</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Started</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {recentGames.map((game) => (
                      <tr key={game.id}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 truncate max-w-xs">
                            {game.name} <span className="text-gray-500">(ID: {game.id})</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <GameStatusBadge status={game.status} />
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">{Array.isArray(game.players) ? game.players.length : game.players ?? 0}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900 truncate max-w-[12rem]">{formatDateTime(game.started_at || game.created_at)}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{humanDuration(game.started_at || game.created_at, game.completed_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Security */}
          {activeTab === 'security' && (
            <div className="card-surface pad-6 rounded-lg">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Security Settings</h3>
              <div className="space-y-4">
                <label className="flex items-center space-x-3 text-sm"><input type="checkbox" defaultChecked className="h-4 w-4" /> <span>Require email verification</span></label>
                <label className="flex items-center space-x-3 text-sm"><input type="checkbox" className="h-4 w-4" /> <span>Enable two-factor authentication</span></label>
                <label className="flex items-center space-x-3 text-sm"><input type="checkbox" defaultChecked className="h-4 w-4" /> <span>Allow password reset</span></label>
              </div>
            </div>
          )}

          {/* Reports */}
          {activeTab === 'reports' && (
            <div className="bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Reports</h3>
                    <p className="mt-1 max-w-2xl text-sm text-gray-500">Generate and download system reports</p>
                  </div>
                  <div className="flex space-x-3">
                    <select id="report-type" name="report-type" className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 sm:text-sm rounded-md" defaultValue="user-activity">
                      <option value="user-activity">User Activity</option>
                      <option value="game-stats">Game Statistics</option>
                      <option value="system-usage">System Usage</option>
                      <option value="financial">Financial Reports</option>
                    </select>
                    <button type="button" className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600">Generate Report</button>
                  </div>
                </div>
              </div>
              <div className="px-4 py-5 sm:p-6">
                <div className="h-48 flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg">
                  <div className="text-center text-sm text-gray-600">Select a report type and click Generate.</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Ranges Modal */}
      <RangesModal
        visible={showRangesModal}
        rangeEdits={rangeEdits}
        originalConfig={(gameConfigs[null] || {}).system_config || {}}
        onClose={() => {
          setShowRangesModal(false);
          // Reset to current config (if editing from a game), otherwise keep existing
        }}
        onChange={(key, next) => setRangeEdits((prev) => ({ ...prev, [key]: next }))}
        onSave={async () => {
          try {
            await mixedGameApi.saveSystemConfig(rangeEdits);
            const games = await mixedGameApi.getGames();
            const map = {};
            (games || []).forEach((g) => {
              map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {} };
            });
            setGameConfigs(map);
            setShowRangesModal(false);
          } catch (error) {
            console.error('Failed to save system config:', error);
          }
        }}
      />
    </>
  );
};

export default AdminDashboard;
