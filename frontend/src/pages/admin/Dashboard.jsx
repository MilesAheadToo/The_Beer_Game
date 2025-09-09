import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  LineElement, 
  PointElement, 
  Title, 
  Tooltip, 
  Legend, 
  ArcElement 
} from 'chart.js';
import { Bar, Line, Pie } from 'react-chartjs-2';
import { 
  UsersIcon, 
  ClockIcon, 
  ChartBarIcon, 
  UserGroupIcon, 
  CogIcon, 
  ShieldCheckIcon,
  DocumentTextIcon,
  UserCircleIcon
} from '@heroicons/react/24/outline';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const AdminDashboard = () => {
  const { isAdmin } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [recentUsers, setRecentUsers] = useState([]);
  const [recentGames, setRecentGames] = useState([]);
  const [gameConfigs, setGameConfigs] = useState({});
  const [showConfigFor, setShowConfigFor] = useState(null);
  const [editing, setEditing] = useState(false);
  const [editingPolicies, setEditingPolicies] = useState({});
  const [editingRanges, setEditingRanges] = useState(false);
  const [rangeEdits, setRangeEdits] = useState({});
  const [showRangesModal, setShowRangesModal] = useState(false);
  const [editingPricing, setEditingPricing] = useState(false);
  const [pricingEdits, setPricingEdits] = useState({});
  const [editingGlobal, setEditingGlobal] = useState(false);
  const [globalEdits, setGlobalEdits] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');
  const [error, setError] = useState(null);

  // Check if user is admin
  useEffect(() => {
    if (!isAdmin) {
      navigate('/unauthorized');
    }
  }, [isAdmin, navigate]);

  // Fetch admin data
  useEffect(() => {
    const fetchAdminData = async () => {
      try {
        setIsLoading(true);
        
        // In a real app, these would be actual API calls
        // For now, we'll use mock data
        const mockStats = {
          totalUsers: 1245,
          activeUsers: 342,
          totalGames: 876,
          activeGames: 23,
          avgGameDuration: 45,
          userGrowth: 12.5,
          gameGrowth: 8.2,
          userActivity: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            data: [65, 59, 80, 81, 56, 55, 40],
          },
          gameStats: {
            completed: 65,
            inProgress: 23,
            abandoned: 12,
          },
          userDistribution: {
            new: 45,
            returning: 30,
            inactive: 25,
          },
        };

        const mockRecentUsers = [
          { id: 1, username: 'johndoe', email: 'john@example.com', joined: '2023-05-15T10:30:00Z', status: 'active' },
          { id: 2, username: 'alice_smith', email: 'alice@example.com', joined: '2023-05-14T15:45:00Z', status: 'active' },
          { id: 3, username: 'bob_johnson', email: 'bob@example.com', joined: '2023-05-13T09:20:00Z', status: 'inactive' },
          { id: 4, username: 'emma_wilson', email: 'emma@example.com', joined: '2023-05-12T14:10:00Z', status: 'active' },
          { id: 5, username: 'michael_brown', email: 'michael@example.com', joined: '2023-05-11T11:05:00Z', status: 'suspended' },
        ];

        const mockRecentGames = [
          { id: 101, name: 'Supply Chain Masters', status: 'completed', players: 4, started: '2023-05-15T09:30:00Z', duration: '25m' },
          { id: 102, name: 'Beer Game Pro', status: 'in_progress', players: 3, started: '2023-05-15T10:15:00Z', duration: '15m' },
          { id: 103, name: 'Logistics Challenge', status: 'completed', players: 4, started: '2023-05-14T16:45:00Z', duration: '32m' },
          { id: 104, name: 'Supply Chain Newbies', status: 'abandoned', players: 2, started: '2023-05-14T18:20:00Z', duration: '8m' },
          { id: 105, name: 'Beer Distribution', status: 'completed', players: 4, started: '2023-05-13T14:10:00Z', duration: '28m' },
        ];

        setStats(mockStats);
        setRecentUsers(mockRecentUsers);
        try {
          const games = await mixedGameApi.getGames();
          setRecentGames(games || []);
          const map = {};
          (games || []).forEach(g => {
            map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {} };
          });
          setGameConfigs(map);
        } catch (e) {
          setRecentGames([]);
        }
        setError(null);
      } catch (err) {
        console.error('Failed to fetch admin data:', err);
        setError('Failed to load admin dashboard. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchAdminData();
  }, []);

  // If navigated with ?openSystemRanges=1, open ranges modal with server config
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

  // Stats card component
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
                {change && (
                  <div className={`ml-2 flex items-baseline text-sm font-medium ${
                    changeType === 'increase' ? 'text-green-600' : 
                    changeType === 'decrease' ? 'text-red-600' : 'text-gray-500'
                  }`}>
                    {changeType === 'increase' ? (
                      <span className="text-green-500">↑</span>
                    ) : changeType === 'decrease' ? (
                      <span className="text-red-500">↓</span>
                    ) : null}
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

  // User activity chart data
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

  // Game status chart data
  const gameStatusData = {
    labels: ['Completed', 'In Progress', 'Abandoned'],
    datasets: [
      {
        data: [
          stats?.gameStats?.completed || 0,
          stats?.gameStats?.inProgress || 0,
          stats?.gameStats?.abandoned || 0,
        ],
        backgroundColor: [
          'rgba(16, 185, 129, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(239, 68, 68, 0.8)',
        ],
        borderColor: [
          'rgba(16, 185, 129, 1)',
          'rgba(59, 130, 246, 1)',
          'rgba(239, 68, 68, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // User distribution chart data
  const userDistributionData = {
    labels: ['New Users', 'Returning Users', 'Inactive Users'],
    datasets: [
      {
        data: [
          stats?.userDistribution?.new || 0,
          stats?.userDistribution?.returning || 0,
          stats?.userDistribution?.inactive || 0,
        ],
        backgroundColor: [
          'rgba(99, 102, 241, 0.8)',
          'rgba(139, 92, 246, 0.8)',
          'rgba(156, 163, 175, 0.8)',
        ],
        borderColor: [
          'rgba(99, 102, 241, 1)',
          'rgba(139, 92, 246, 1)',
          'rgba(156, 163, 175, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          precision: 0,
        },
      },
    },
  };

  // User status badge
  const UserStatusBadge = ({ status }) => {
    const statusMap = {
      active: { text: 'Active', color: 'bg-green-100 text-green-800' },
      inactive: { text: 'Inactive', color: 'bg-yellow-100 text-yellow-800' },
      suspended: { text: 'Suspended', color: 'bg-red-100 text-red-800' },
      pending: { text: 'Pending', color: 'bg-blue-100 text-blue-800' },
    };

    const statusInfo = statusMap[status.toLowerCase()] || { text: status, color: 'bg-gray-100 text-gray-800' };

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.color}`}>
        {statusInfo.text}
      </span>
    );
  };

  // Game status badge
  const GameStatusBadge = ({ status }) => {
    const statusMap = {
      completed: { text: 'Completed', color: 'bg-green-100 text-green-800' },
      in_progress: { text: 'In Progress', color: 'bg-blue-100 text-blue-800' },
      abandoned: { text: 'Abandoned', color: 'bg-red-100 text-red-800' },
      waiting: { text: 'Waiting', color: 'bg-yellow-100 text-yellow-800' },
    };

    const statusInfo = statusMap[status] || { text: status, color: 'bg-gray-100 text-gray-800' };

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusInfo.color}`}>
        {statusInfo.text}
      </span>
    );
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/4"></div>
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="card-surface pad-6 rounded-lg h-32">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
                <div className="h-8 bg-gray-200 rounded w-1/2"></div>
              </div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card-surface pad-6 rounded-lg h-80"></div>
            <div className="card-surface pad-6 rounded-lg h-80"></div>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="card-surface">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
            <div className="flex space-x-3">
              <button
                type="button"
                onClick={() => navigate('/users')}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                <UsersIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                Users
              </button>
              <button
                type="button"
                onClick={() => setShowRangesModal(true)}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <CogIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                Edit Ranges
              </button>
              <button
                type="button"
                onClick={() => navigate('/admin/training')}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
              >
                Training
              </button>
            </div>
          </div>
          
          {/* Tabs */}
          <div className="mt-6">
            <div className="border-b border-gray-200">
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
                    className={`${activeTab === tab.id
                      ? 'border-indigo-500 text-indigo-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}
                  >
                    <tab.icon className="mr-2 h-5 w-5" />
                    {tab.name}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {/* Stats Overview */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Stats Grid */}
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
              <StatCard 
                title="Total Users" 
                value={stats?.totalUsers?.toLocaleString() || '0'} 
                icon={UsersIcon}
                change={stats?.userGrowth}
                changeType={stats?.userGrowth >= 0 ? 'increase' : 'decrease'}
                loading={isLoading}
              />
              <StatCard 
                title="Active Users" 
                value={stats?.activeUsers?.toLocaleString() || '0'} 
                icon={UserCircleIcon}
                loading={isLoading}
              />
              <StatCard 
                title="Total Games" 
                value={stats?.totalGames?.toLocaleString() || '0'} 
                icon={UserGroupIcon}
                change={stats?.gameGrowth}
                changeType={stats?.gameGrowth >= 0 ? 'increase' : 'decrease'}
                loading={isLoading}
              />
              <StatCard 
                title="Active Games" 
                value={stats?.activeGames?.toLocaleString() || '0'} 
                icon={ClockIcon}
                loading={isLoading}
              />
            </div>

            {/* Charts Row 1 */}
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

            {/* Recent Activity */}
            <div className="card-surface overflow-hidden sm:rounded-lg">
              <div className="pad-6 border-b border-gray-200">
                <h3 className="text-lg leading-6 font-medium text-gray-900">Recent Activity</h3>
                <p className="mt-1 max-w-2xl text-sm text-gray-500">Latest actions and events in the system</p>
              </div>
              <div className="overflow-hidden">
                <ul className="divide-y divide-gray-200">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <li key={i} className="pad-6">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-indigo-600 truncate">
                          New user registered: user{Math.floor(Math.random() * 1000)}
                        </p>
                        <div className="ml-2 flex-shrink-0 flex">
                          <p className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            {Math.floor(Math.random() * 60)}m ago
                          </p>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Users Tab */}
        {activeTab === 'users' && (
          <div className="card-surface overflow-hidden sm:rounded-lg">
            <div className="pad-6 border-b border-gray-200">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-lg leading-6 font-medium text-gray-900">User Management</h3>
                  <p className="mt-1 max-w-2xl text-sm text-gray-500">Manage system users and permissions</p>
                </div>
                <button
                  type="button"
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Add User
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      User
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Email
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Joined
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th scope="col" className="relative px-6 py-3">
                      <span className="sr-only">Actions</span>
                    </th>
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
                        <div className="text-sm text-gray-900">
                          {new Date(user.joined).toLocaleDateString()}
                        </div>
                        <div className="text-sm text-gray-500">
                          {new Date(user.joined).toLocaleTimeString()}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <UserStatusBadge status={user.status} />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button className="text-indigo-600 hover:text-indigo-900 mr-4">Edit</button>
                        <button className="text-red-600 hover:text-red-900">Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
              <div className="flex-1 flex justify-between sm:hidden">
                <button className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                  Previous
                </button>
                <button className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                  Next
                </button>
              </div>
              <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm text-gray-700">
                    Showing <span className="font-medium">1</span> to <span className="font-medium">5</span> of{' '}
                    <span className="font-medium">24</span> results
                  </p>
                </div>
                <div>
                  <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                    <button className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                      <span className="sr-only">Previous</span>
                      <span className="h-5 w-5">«</span>
                    </button>
                    <button className="bg-indigo-50 border-indigo-500 text-indigo-600 relative inline-flex items-center px-4 py-2 border text-sm font-medium">
                      1
                    </button>
                    <button className="bg-white border-gray-300 text-gray-500 hover:bg-gray-50 relative inline-flex items-center px-4 py-2 border text-sm font-medium">
                      2
                    </button>
                    <button className="bg-white border-gray-300 text-gray-500 hover:bg-gray-50 relative inline-flex items-center px-4 py-2 border text-sm font-medium">
                      3
                    </button>
                    <button className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                      <span className="sr-only">Next</span>
                      <span className="h-5 w-5">»</span>
                    </button>
                  </nav>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Games Tab */}
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
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Game
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Players
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Started
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th scope="col" className="relative px-6 py-3">
                      <span className="sr-only">Actions</span>
                    </th>
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
                        <div className="text-sm text-gray-900">{game.players}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900 truncate max-w-[12rem]">
                          {new Date(game.started).toLocaleDateString()} {new Date(game.started).toLocaleTimeString()}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {game.duration || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-indigo-600">
                        <button onClick={() => {
                          const next = showConfigFor === game.id ? null : game.id;
                          setShowConfigFor(next);
                          setEditing(false);
                          if (next) setEditingPolicies(JSON.parse(JSON.stringify((gameConfigs[game.id]||{}).node_policies||{})));
                          if (next) setRangeEdits(JSON.parse(JSON.stringify((gameConfigs[game.id]||{}).system_config||{})));
                          if (next) setPricingEdits(JSON.parse(JSON.stringify((gameConfigs[game.id]||{}).pricing_config||{})));
                          if (next) setGlobalEdits(JSON.parse(JSON.stringify((gameConfigs[game.id]||{}).global_policy||{})));
                        }} className="underline">
                          View Config
                        </button>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button className="text-indigo-600 hover:text-indigo-900 mr-4">View</button>
                        <button className="text-red-600 hover:text-red-900">Delete</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {showConfigFor && (
              <div className="pad-6 border-t border-gray-200">
                <h4 className="text-sm font-medium text-gray-900 mb-4">Configuration for Game #{showConfigFor}</h4>
                <div className="flex justify-end mb-4">
                  {!editing ? (
                    <button className="text-indigo-600 hover:text-indigo-800 text-sm" onClick={() => setEditing(true)}>Edit Policies</button>
                  ) : (
                    <div className="space-x-3">
                      <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setEditingPolicies(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).node_policies||{}))); }}>Revert</button>
                      <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setEditing(false); setEditingPolicies(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).node_policies||{}))); }}>Cancel</button>
                      <button className="text-white bg-indigo-600 px-3 py-1 rounded text-sm" onClick={async () => {
                        await mixedGameApi.updateMixedGame(showConfigFor, { node_policies: editingPolicies });
                        const games = await mixedGameApi.getGames();
                        const map = {};
                        (games || []).forEach(g => { map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {}, pricing_config: g.pricing_config || {}, global_policy: g.global_policy || {} }; });
                        setGameConfigs(map);
                        setEditing(false);
                      }}>Save</button>
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Node Policies */}
                  <div className="card-surface pad-6">
                    <h5 className="text-sm font-semibold text-gray-900 mb-2">Per-Node Policies</h5>
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Node</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Order LT</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Supply LT</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Init Inv</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Min Ord</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Price</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Std Cost</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Var Cost</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {Object.entries(editing ? editingPolicies : (gameConfigs[showConfigFor] || {}).node_policies || {}).map(([node, pol]) => (
                          <tr key={node}>
                            <td className="px-3 py-2 text-sm">{node}</td>
                            {['info_delay','ship_delay','init_inventory','min_order_qty','price','standard_cost','variable_cost'].map((k) => (
                              <td key={k} className="px-3 py-2 text-sm">
                                {!editing ? (
                                  <span>{pol[k]}</span>
                                ) : (
                                  <input type="number" value={editingPolicies[node]?.[k] ?? ''} onChange={(e) => setEditingPolicies(prev => ({...prev, [node]: { ...(prev[node]||{}), [k]: e.target.valueAsNumber }}))} className="w-24 border border-gray-300 rounded px-2 py-1 text-sm" />
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* System Ranges */}
                  <div className="card-surface pad-6">
                    <div className="flex justify-between items-center mb-2">
                      <h5 className="text-sm font-semibold text-gray-900">System Ranges</h5>
                      {!editingRanges ? (
                        <div className="space-x-3">
                          <button className="text-indigo-600 hover:text-indigo-800 text-sm" onClick={() => setEditingRanges(true)}>Quick Edit</button>
                          <button className="text-indigo-600 hover:text-indigo-800 text-sm" onClick={() => setShowRangesModal(true)}>Open Editor</button>
                        </div>
                      ) : (
                        <div className="space-x-3">
                          <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setRangeEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).system_config||{}))); }}>Revert</button>
                          <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setEditingRanges(false); setRangeEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).system_config||{}))); }}>Cancel</button>
                          <button className="text-white bg-indigo-600 px-3 py-1 rounded text-sm" onClick={async () => {
                            await mixedGameApi.saveSystemConfig(rangeEdits);
                            const games = await mixedGameApi.getGames();
                            const map = {};
                            (games || []).forEach(g => { map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {}, pricing_config: g.pricing_config || {}, global_policy: g.global_policy || {} }; });
                            setGameConfigs(map);
                            setEditingRanges(false);
                          }}>Save</button>
                        </div>
                      )}
                    </div>
                    {!editingRanges ? (
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        {Object.entries((gameConfigs[showConfigFor] || {}).system_config || {}).map(([k, rng]) => (
                          <div key={k} className="flex items-center justify-between bg-gray-50 rounded px-3 py-2">
                            <span className="text-gray-600 truncate mr-2">{k.replaceAll('_',' ')}</span>
                            <span className="font-medium">{rng.min} – {rng.max}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        {Object.entries(rangeEdits || {}).map(([k, rng]) => (
                          <div key={k} className="bg-gray-50 rounded px-3 py-2 flex items-center justify-between">
                            <span className="text-gray-600 truncate mr-2">{k.replaceAll('_',' ')}</span>
                            <span className="space-x-2">
                              <input type="number" className="w-20 border border-gray-300 rounded px-2 py-1 text-sm" value={rng.min} onChange={e => setRangeEdits(prev => ({...prev, [k]: { ...prev[k], min: e.target.valueAsNumber }}))} />
                              <input type="number" className="w-20 border border-gray-300 rounded px-2 py-1 text-sm" value={rng.max} onChange={e => setRangeEdits(prev => ({...prev, [k]: { ...prev[k], max: e.target.valueAsNumber }}))} />
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Pricing Config + Global Policy */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                  <div className="card-surface pad-6">
                    <div className="flex justify-between items-center mb-2">
                      <h5 className="text-sm font-semibold text-gray-900">Pricing Config</h5>
                      {!editingPricing ? (
                        <button className="text-indigo-600 hover:text-indigo-800 text-sm" onClick={() => setEditingPricing(true)}>Edit Pricing</button>
                      ) : (
                        <div className="space-x-3">
                          <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setPricingEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).pricing_config||{}))); }}>Revert</button>
                          <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setEditingPricing(false); setPricingEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).pricing_config||{}))); }}>Cancel</button>
                          <button className="text-white bg-indigo-600 px-3 py-1 rounded text-sm" onClick={async () => {
                            await mixedGameApi.updateMixedGame(showConfigFor, { pricing_config: pricingEdits });
                            const games = await mixedGameApi.getGames();
                            const map = {};
                            (games || []).forEach(g => { map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {}, pricing_config: g.pricing_config || {}, global_policy: g.global_policy || {} }; });
                            setGameConfigs(map);
                            setEditingPricing(false);
                          }}>Save</button>
                        </div>
                      )}
                    </div>
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Role</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Price</th>
                          <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Std Cost</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {Object.entries(editingPricing ? pricingEdits : (gameConfigs[showConfigFor] || {}).pricing_config || {}).map(([role, cfg]) => (
                          <tr key={role}>
                            <td className="px-3 py-2 text-sm capitalize">{role}</td>
                            <td className="px-3 py-2 text-sm">
                              {!editingPricing ? (
                                <span>{cfg.selling_price}</span>
                              ) : (
                                <input type="number" className="w-24 border border-gray-300 rounded px-2 py-1 text-sm" value={pricingEdits[role]?.selling_price ?? ''} onChange={e => setPricingEdits(prev => ({...prev, [role]: { ...(prev[role]||{}), selling_price: e.target.valueAsNumber }}))} />
                              )}
                            </td>
                            <td className="px-3 py-2 text-sm">
                              {!editingPricing ? (
                                <span>{cfg.standard_cost}</span>
                              ) : (
                                <input type="number" className="w-24 border border-gray-300 rounded px-2 py-1 text-sm" value={pricingEdits[role]?.standard_cost ?? ''} onChange={e => setPricingEdits(prev => ({...prev, [role]: { ...(prev[role]||{}), standard_cost: e.target.valueAsNumber }}))} />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="card-surface pad-6">
                    <div className="flex justify-between items-center mb-2">
                      <h5 className="text-sm font-semibold text-gray-900">Global Policy</h5>
                      {!editingGlobal ? (
                        <button className="text-indigo-600 hover:text-indigo-800 text-sm" onClick={() => setEditingGlobal(true)}>Edit Global</button>
                      ) : (
                        <div className="space-x-3">
                          <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setGlobalEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).global_policy||{}))); }}>Revert</button>
                          <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setEditingGlobal(false); setGlobalEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).global_policy||{}))); }}>Cancel</button>
                          <button className="text-white bg-indigo-600 px-3 py-1 rounded text-sm" onClick={async () => {
                            await mixedGameApi.updateMixedGame(showConfigFor, { global_policy: globalEdits });
                            const games = await mixedGameApi.getGames();
                            const map = {};
                            (games || []).forEach(g => { map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {}, pricing_config: g.pricing_config || {}, global_policy: g.global_policy || {} }; });
                            setGameConfigs(map);
                            setEditingGlobal(false);
                          }}>Save</button>
                        </div>
                      )}
                    </div>
                    {!editingGlobal ? (
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        {Object.entries((gameConfigs[showConfigFor] || {}).global_policy || {}).map(([k, v]) => (
                          <div key={k} className="flex items-center justify-between bg-gray-50 rounded px-3 py-2">
                            <span className="text-gray-600 truncate mr-2">{k.replaceAll('_',' ')}</span>
                            <span className="font-medium">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        {Object.entries(globalEdits || {}).map(([k, v]) => (
                          <div key={k} className="bg-gray-50 rounded px-3 py-2 flex items-center justify-between">
                            <span className="text-gray-600 truncate mr-2">{k.replaceAll('_',' ')}</span>
                            <input type="number" className="w-24 border border-gray-300 rounded px-2 py-1 text-sm" value={globalEdits[k] ?? ''} onChange={e => setGlobalEdits(prev => ({...prev, [k]: e.target.valueAsNumber}))} />
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Ranges Modal */}
              {showRangesModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
                  <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl">
                    <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
                      <h4 className="text-base font-semibold">Edit System Ranges</h4>
                      <button className="text-gray-500 hover:text-gray-700" onClick={() => { setShowRangesModal(false); setRangeEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).system_config||{}))); }}>✕</button>
                    </div>
                    <div className="p-4 max-h-[70vh] overflow-auto">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(rangeEdits || {}).map(([k, rng]) => {
                          const orig = ((gameConfigs[showConfigFor]||{}).system_config||{})[k] || {};
                          const changed = (rng?.min !== orig?.min) || (rng?.max !== orig?.max);
                          const invalid = Number.isFinite(rng?.min) && Number.isFinite(rng?.max) && rng.min > rng.max;
                          return (
                            <div key={k} className={`rounded border ${invalid? 'border-red-300 bg-red-50' : changed ? 'border-yellow-300 bg-yellow-50' : 'border-gray-200 bg-gray-50'} p-3`}>
                              <div className="text-sm text-gray-700 mb-2 capitalize">{k.replaceAll('_',' ')}</div>
                              <div className="flex items-center space-x-2">
                                <div className="flex-1">
                                  <label className="block text-xs text-gray-500 mb-1">Min</label>
                                  <input type="number" className="w-full border border-gray-300 rounded px-2 py-1 text-sm" value={rng.min} onChange={(e)=> setRangeEdits(prev => ({...prev, [k]: { ...prev[k], min: e.target.valueAsNumber }}))} />
                                </div>
                                <div className="flex-1">
                                  <label className="block text-xs text-gray-500 mb-1">Max</label>
                                  <input type="number" className="w-full border border-gray-300 rounded px-2 py-1 text-sm" value={rng.max} onChange={(e)=> setRangeEdits(prev => ({...prev, [k]: { ...prev[k], max: e.target.valueAsNumber }}))} />
                                </div>
                              </div>
                              {invalid && <div className="mt-1 text-xs text-red-600">Min must be ≤ Max</div>}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                    <div className="px-4 py-3 border-t border-gray-200 flex justify-end space-x-2">
                      <button className="text-gray-600 hover:text-gray-800 text-sm" onClick={() => { setShowRangesModal(false); setRangeEdits(JSON.parse(JSON.stringify((gameConfigs[showConfigFor]||{}).system_config||{}))); }}>Cancel</button>
                      <button className="text-white bg-indigo-600 px-3 py-1 rounded text-sm disabled:opacity-50" disabled={Object.values(rangeEdits||{}).some(r => (r?.min ?? 0) > (r?.max ?? 0))} onClick={async () => {
                        await mixedGameApi.saveSystemConfig(rangeEdits);
                        const games = await mixedGameApi.getGames();
                        const map = {};
                        (games || []).forEach(g => { map[g.id] = { node_policies: g.node_policies || {}, system_config: g.system_config || {}, pricing_config: g.pricing_config || {}, global_policy: g.global_policy || {} }; });
                        setGameConfigs(map);
                        setShowRangesModal(false);
                      }}>Save</button>
                    </div>
                  </div>
                </div>
              )}
            )}
            <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
              <div className="flex-1 flex justify-between sm:hidden">
                <button className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                  Previous
                </button>
                <button className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                  Next
                </button>
              </div>
              <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm text-gray-700">
                    Showing <span className="font-medium">1</span> to <span className="font-medium">5</span> of{' '}
                    <span className="font-medium">12</span> results
                  </p>
                </div>
                <div>
                  <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                    <button className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                      <span className="sr-only">Previous</span>
                      <span className="h-5 w-5">«</span>
                    </button>
                    <button className="bg-indigo-50 border-indigo-500 text-indigo-600 relative inline-flex items-center px-4 py-2 border text-sm font-medium">
                      1
                    </button>
                    <button className="bg-white border-gray-300 text-gray-500 hover:bg-gray-50 relative inline-flex items-center px-4 py-2 border text-sm font-medium">
                      2
                    </button>
                    <button className="bg-white border-gray-300 text-gray-500 hover:bg-gray-50 relative inline-flex items-center px-4 py-2 border text-sm font-medium">
                      3
                    </button>
                    <button className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50">
                      <span className="sr-only">Next</span>
                      <span className="h-5 w-5">»</span>
                    </button>
                  </nav>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Security Tab */}
        {activeTab === 'security' && (
          <div className="card-surface overflow-hidden sm:rounded-lg">
            <div className="pad-6 border-b border-gray-200">
              <h3 className="text-lg leading-6 font-medium text-gray-900">Security Settings</h3>
              <p className="mt-1 max-w-2xl text-sm text-gray-500">Manage security settings and access controls</p>
            </div>
            <div className="pad-6">
              <div className="space-y-6">
                <div className="md:grid md:grid-cols-3 md:gap-6">
                  <div className="md:col-span-1">
                    <h3 className="text-lg font-medium leading-6 text-gray-900">Authentication</h3>
                    <p className="mt-1 text-sm text-gray-500">Configure how users authenticate to the system.</p>
                  </div>
                  <div className="mt-5 md:mt-0 md:col-span-2">
                    <div className="space-y-6">
                      <div className="flex items-start">
                        <div className="flex items-center h-5">
                          <input
                            id="email-verification"
                            name="email-verification"
                            type="checkbox"
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                            defaultChecked
                          />
                        </div>
                        <div className="ml-3 text-sm">
                          <label htmlFor="email-verification" className="font-medium text-gray-700">
                            Require email verification
                          </label>
                          <p className="text-gray-500">Users must verify their email address before they can log in.</p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="flex items-center h-5">
                          <input
                            id="two-factor"
                            name="two-factor"
                            type="checkbox"
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                        </div>
                        <div className="ml-3 text-sm">
                          <label htmlFor="two-factor" className="font-medium text-gray-700">
                            Enable Two-Factor Authentication
                          </label>
                          <p className="text-gray-500">Users will be required to set up 2FA for their accounts.</p>
                        </div>
                      </div>
                      <div className="flex items-start">
                        <div className="flex items-center h-5">
                          <input
                            id="password-reset"
                            name="password-reset"
                            type="checkbox"
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300 rounded"
                            defaultChecked
                          />
                        </div>
                        <div className="ml-3 text-sm">
                          <label htmlFor="password-reset" className="font-medium text-gray-700">
                            Allow password reset
                          </label>
                          <p className="text-gray-500">Users can reset their password via email.</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="pt-5 border-t border-gray-200">
                  <div className="flex justify-end">
                    <button
                      type="button"
                      className="bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      Save
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Reports Tab */}
        {activeTab === 'reports' && (
          <div className="bg-white shadow overflow-hidden sm:rounded-lg">
            <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Reports</h3>
                  <p className="mt-1 max-w-2xl text-sm text-gray-500">Generate and download system reports</p>
                </div>
                <div className="flex space-x-3">
                  <select
                    id="report-type"
                    name="report-type"
                    className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                    defaultValue="user-activity"
                  >
                    <option value="user-activity">User Activity</option>
                    <option value="game-stats">Game Statistics</option>
                    <option value="system-usage">System Usage</option>
                    <option value="financial">Financial Reports</option>
                  </select>
                  <button
                    type="button"
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Generate Report
                  </button>
                </div>
              </div>
            </div>
            <div className="px-4 py-5 sm:p-6">
              <div className="h-96 flex items-center justify-center border-2 border-dashed border-gray-300 rounded-lg">
                <div className="text-center">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No report selected</h3>
                  <p className="mt-1 text-sm text-gray-500">
                    Select a report type and click "Generate Report" to view data.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AdminDashboard;
