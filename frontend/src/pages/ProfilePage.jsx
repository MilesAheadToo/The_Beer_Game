import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { 
  UserCircleIcon, 
  TrophyIcon, 
  ChartBarIcon, 
  CalendarIcon,
  PencilIcon,
  CheckIcon,
  XMarkIcon,
  ShieldCheckIcon,
  
} from '@heroicons/react/24/outline';

const ProfilePage = () => {
  const { user, updateProfile } = useAuth();
  const [profile, setProfile] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    bio: '',
    avatar: ''
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch user profile and leaderboard data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // In a real app, these would be actual API calls
        // For now, we'll use mock data
        const mockProfile = {
          id: user?.id,
          username: user?.username || 'johndoe',
          email: user?.email || 'john@example.com',
          bio: 'Supply chain enthusiast and beer game master. Always looking for new challenges!',
          avatar: user?.avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(user?.username || 'User')}&background=random`,
          joinDate: '2023-01-15T10:30:00Z',
          stats: {
            gamesPlayed: 42,
            gamesWon: 28,
            winRate: 67,
            averageScore: 1245,
            totalPlayTime: '3d 7h 22m',
            currentStreak: 5,
            highestStreak: 8,
            rank: 7,
            totalPlayers: 1245,
          },
          achievements: [
            { id: 1, name: 'First Win', description: 'Win your first game', icon: '🏆', earned: true, date: '2023-01-20' },
            { id: 2, name: 'Supply Chain Master', description: 'Win 25 games', icon: '🏅', earned: true, date: '2023-03-15' },
            { id: 3, name: 'Perfect Game', description: 'Win a game with maximum score', icon: '⭐', earned: false },
            { id: 4, name: 'Marathoner', description: 'Play for more than 10 hours', icon: '⏱️', earned: true, date: '2023-04-02' },
            { id: 5, name: 'Social Butterfly', description: 'Play with 50 different players', icon: '🦋', earned: false },
          ],
          recentGames: [
            { id: 101, name: 'Supply Chain Masters', status: 'won', score: 1450, position: 1, date: '2023-05-15T14:30:00Z' },
            { id: 102, name: 'Beer Distribution', status: 'lost', score: 980, position: 3, date: '2023-05-14T10:15:00Z' },
            { id: 103, name: 'Logistics Challenge', status: 'won', score: 1320, position: 2, date: '2023-05-12T16:45:00Z' },
            { id: 104, name: 'Supply Chain Newbies', status: 'won', score: 1560, position: 1, date: '2023-05-10T09:20:00Z' },
            { id: 105, name: 'Beer Game Pro', status: 'lost', score: 890, position: 4, date: '2023-05-08T11:30:00Z' },
          ]
        };

        const mockLeaderboard = [
          { id: 1, username: 'supplychainmaster', score: 24560, gamesPlayed: 87, winRate: 82, avatar: 'https://i.pravatar.cc/150?img=1' },
          { id: 2, username: 'beerbaron', score: 23120, gamesPlayed: 92, winRate: 78, avatar: 'https://i.pravatar.cc/150?img=2' },
          { id: 3, username: 'logisticspro', score: 21890, gamesPlayed: 76, winRate: 81, avatar: 'https://i.pravatar.cc/150?img=3' },
          { id: 4, username: 'inventoryguru', score: 20560, gamesPlayed: 68, winRate: 85, avatar: 'https://i.pravatar.cc/150?img=4' },
          { id: 5, username: 'supplyqueen', score: 19870, gamesPlayed: 72, winRate: 79, avatar: 'https://i.pravatar.cc/150?img=5' },
          { id: 6, username: 'beerwizard', score: 18730, gamesPlayed: 65, winRate: 76, avatar: 'https://i.pravatar.cc/150?img=6' },
          { id: 7, username: profile?.username || 'johndoe', score: profile?.stats?.averageScore * 10 || 12450, gamesPlayed: profile?.stats?.gamesPlayed || 42, winRate: profile?.stats?.winRate || 67, avatar: profile?.avatar, isCurrentUser: true },
          { id: 8, username: 'logisticsninja', score: 12340, gamesPlayed: 51, winRate: 72, avatar: 'https://i.pravatar.cc/150?img=7' },
          { id: 9, username: 'supplychainnewbie', score: 11890, gamesPlayed: 48, winRate: 68, avatar: 'https://i.pravatar.cc/150?img=8' },
          { id: 10, username: 'beerlover', score: 10980, gamesPlayed: 43, winRate: 65, avatar: 'https://i.pravatar.cc/150?img=9' },
        ];

        setProfile(mockProfile);
        setLeaderboard(mockLeaderboard);
        setFormData({
          username: mockProfile.username,
          email: mockProfile.email,
          bio: mockProfile.bio,
          avatar: mockProfile.avatar
        });
        setError(null);
      } catch (err) {
        console.error('Failed to fetch profile data:', err);
        setError('Failed to load profile data. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    if (user) {
      fetchData();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setIsLoading(true);
      // In a real app, this would be an API call to update the profile
      // await gameApi.updateProfile(formData);
      
      // Update local state
      setProfile(prev => ({
        ...prev,
        ...formData
      }));
      
      // Update auth context
      await updateProfile(formData);
      
      setIsEditing(false);
    } catch (err) {
      console.error('Failed to update profile:', err);
      setError('Failed to update profile. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAvatarUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // In a real app, you would upload the file to your server
    // and get back a URL to the uploaded image
    // const formData = new FormData();
    // formData.append('avatar', file);
    // const response = await gameApi.uploadAvatar(formData);
    
    // For demo purposes, we'll just create a local URL
    const imageUrl = URL.createObjectURL(file);
    
    setFormData(prev => ({
      ...prev,
      avatar: imageUrl
    }));
  };

  // Loading state
  if (isLoading || !profile) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse space-y-6">
            <div className="h-8 bg-gray-200 rounded w-1/4"></div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-1 space-y-6">
                <div className="bg-white p-6 rounded-lg shadow h-96"></div>
              </div>
              <div className="md:col-span-2 space-y-6">
                <div className="bg-white p-6 rounded-lg shadow h-96"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-7xl mx-auto">
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
      </div>
    );
  }

  // Format date
  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="md:flex md:items-center md:justify-between mb-8">
          <div className="flex-1 min-w-0">
            <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
              {isEditing ? 'Edit Profile' : 'My Profile'}
            </h2>
            <p className="mt-1 text-sm text-gray-500">
              {isEditing 
                ? 'Update your profile information'
                : 'View and manage your profile, stats, and achievements'}
            </p>
          </div>
          <div className="mt-4 flex md:mt-0 md:ml-4
          ">
            {!isEditing ? (
              <button
                type="button"
                onClick={() => setIsEditing(true)}
                className="ml-3 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                <PencilIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                Edit Profile
              </button>
            ) : (
              <div className="flex space-x-3">
                <button
                  type="button"
                  onClick={() => {
                    setIsEditing(false);
                    // Reset form data
                    setFormData({
                      username: profile.username,
                      email: profile.email,
                      bio: profile.bio,
                      avatar: profile.avatar
                    });
                  }}
                  className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  <XMarkIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isLoading}
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                >
                  <CheckIcon className="-ml-1 mr-2 h-5 w-5" aria-hidden="true" />
                  {isLoading ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 mb-8 pad-6">
          <nav className="-mb-px flex space-x-8">
            {[
              { name: 'Overview', id: 'overview' },
              { name: 'Statistics', id: 'stats' },
              { name: 'Achievements', id: 'achievements' },
              { name: 'Game History', id: 'history' },
              { name: 'Leaderboard', id: 'leaderboard' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`${activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Profile Card */}
          <div className="lg:col-span-1">
            <div className="card-surface overflow-hidden rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex flex-col items-center">
                  <div className="relative">
                    {isEditing ? (
                      <div className="group relative">
                        <img
                          className="h-32 w-32 rounded-full object-cover"
                          src={formData.avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(formData.username)}&background=random`}
                          alt=""
                        />
                        <label 
                          htmlFor="avatar-upload"
                          className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-full opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                        >
                          <PencilIcon className="h-6 w-6 text-white" />
                          <input
                            id="avatar-upload"
                            name="avatar-upload"
                            type="file"
                            className="sr-only"
                            accept="image/*"
                            onChange={handleAvatarUpload}
                          />
                        </label>
                      </div>
                    ) : (
                      <img
                        className="h-32 w-32 rounded-full object-cover"
                        src={profile.avatar}
                        alt=""
                      />
                    )}
                  </div>
                  
                  {isEditing ? (
                    <div className="mt-4 w-full">
                      <div className="mb-4">
                        <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                          Username
                        </label>
                        <input
                          type="text"
                          name="username"
                          id="username"
                          value={formData.username}
                          onChange={handleInputChange}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                        />
                      </div>
                      <div className="mb-4">
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                          Email
                        </label>
                        <input
                          type="email"
                          name="email"
                          id="email"
                          value={formData.email}
                          onChange={handleInputChange}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                        />
                      </div>
                      <div>
                        <label htmlFor="bio" className="block text-sm font-medium text-gray-700">
                          Bio
                        </label>
                        <textarea
                          id="bio"
                          name="bio"
                          rows="3"
                          value={formData.bio}
                          onChange={handleInputChange}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="text-center mt-4">
                      <h3 className="text-lg leading-6 font-medium text-gray-900">{profile.username}</h3>
                      <p className="mt-1 text-sm text-gray-500">{profile.bio}</p>
                      <div className="mt-4 flex items-center justify-center text-sm text-gray-500">
                        <CalendarIcon className="flex-shrink-0 mr-1.5 h-5 w-5 text-gray-400" />
                        Member since {formatDate(profile.joinDate)}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              {!isEditing && (
                <div className="bg-gray-50 px-4 py-4 sm:px-6">
                  <div className="flex flex-wrap justify-center gap-4">
                    <div className="text-center">
                      <p className="text-sm font-medium text-gray-500">Rank</p>
                      <p className="text-lg font-semibold text-gray-900">#{profile.stats.rank}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-gray-500">Games</p>
                      <p className="text-lg font-semibold text-gray-900">{profile.stats.gamesPlayed}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-gray-500">Win Rate</p>
                      <p className="text-lg font-semibold text-gray-900">{profile.stats.winRate}%</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Quick Stats */}
            {!isEditing && activeTab !== 'leaderboard' && (
              <div className="mt-6 card-surface overflow-hidden rounded-lg">
                <div className="pad-6">
                  <h3 className="text-lg font-medium text-gray-900">Quick Stats</h3>
                  <dl className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2">
                    <div className="px-4 py-5 bg-gray-50 rounded-lg overflow-hidden sm:p-6">
                      <dt className="text-sm font-medium text-gray-500 truncate">Highest Score</dt>
                      <dd className="mt-1 text-3xl font-semibold text-gray-900">1,890</dd>
                    </div>
                    <div className="px-4 py-5 bg-gray-50 rounded-lg overflow-hidden sm:p-6">
                      <dt className="text-sm font-medium text-gray-500 truncate">Current Streak</dt>
                      <dd className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.currentStreak} days</dd>
                    </div>
                    <div className="px-4 py-5 bg-gray-50 rounded-lg overflow-hidden sm:p-6">
                      <dt className="text-sm font-medium text-gray-500 truncate">Total Play Time</dt>
                      <dd className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.totalPlayTime}</dd>
                    </div>
                    <div className="px-4 py-5 bg-gray-50 rounded-lg overflow-hidden sm:p-6">
                      <dt className="text-sm font-medium text-gray-500 truncate">Achievements</dt>
                      <dd className="mt-1 text-3xl font-semibold text-gray-900">
                        {profile.achievements.filter(a => a.earned).length}/{profile.achievements.length}
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Content */}
          <div className="lg:col-span-2">
            {activeTab === 'overview' && !isEditing && (
              <div className="space-y-6">
                {/* Welcome Card */}
                <div className="bg-indigo-50 rounded-lg p-6">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <UserCircleIcon className="h-12 w-12 text-indigo-400" aria-hidden="true" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-lg font-medium text-indigo-800">Welcome back, {profile.username}!</h3>
                      <div className="mt-2 text-sm text-indigo-700">
                        <p>You're currently ranked <span className="font-semibold">#{profile.stats.rank}</span> out of {profile.stats.totalPlayers} players.</p>
                        <p className="mt-1">You've played {profile.stats.gamesPlayed} games with a {profile.stats.winRate}% win rate.</p>
                      </div>
                      <div className="mt-4">
                        <button
                          type="button"
                          onClick={() => setActiveTab('leaderboard')}
                          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        >
                          View Leaderboard
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Activity */}
                <div className="card-surface overflow-hidden sm:rounded-lg">
                  <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Recent Activity</h3>
                  </div>
                  <div className="bg-white overflow-hidden">
                    <ul className="divide-y divide-gray-200">
                      {profile.recentGames.map((game) => (
                        <li key={game.id} className="px-4 py-4 sm:px-6">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center">
                              <div className="flex-shrink-0">
                                {game.status === 'won' ? (
                                  <TrophyIcon className="h-8 w-8 text-yellow-500" />
                                ) : (
                                  <ChartBarIcon className="h-8 w-8 text-gray-400" />
                                )}
                              </div>
                              <div className="ml-4">
                                <p className="text-sm font-medium text-gray-900">
                                  {game.status === 'won' ? 'You won' : 'You played'} <span className="font-semibold">{game.name}</span>
                                </p>
                                <p className="text-sm text-gray-500">
                                  Scored {game.score} points • {new Date(game.date).toLocaleDateString()}
                                </p>
                              </div>
                            </div>
                            <div className="ml-4 flex-shrink-0">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                game.status === 'won' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                              }`}>
                                {game.status === 'won' ? 'Victory' : 'Completed'}
                              </span>
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="bg-gray-50 px-4 py-4 sm:px-6 text-right text-sm font-medium">
                    <button
                      onClick={() => setActiveTab('history')}
                      className="text-indigo-600 hover:text-indigo-900"
                    >
                      View all activity →
                    </button>
                  </div>
                </div>

                {/* Next Achievement */}
                <div className="card-surface overflow-hidden sm:rounded-lg">
                  <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Next Achievement</h3>
                  </div>
                  <div className="px-4 py-5 sm:p-6">
                    {profile.achievements.find(a => !a.earned) ? (
                      <div className="flex items-start">
                        <div className="flex-shrink-0 bg-yellow-100 rounded-full p-3">
                          <span className="text-2xl">
                            {profile.achievements.find(a => !a.earned).icon}
                          </span>
                        </div>
                        <div className="ml-4">
                          <h4 className="text-lg font-medium text-gray-900">
                            {profile.achievements.find(a => !a.earned).name}
                          </h4>
                          <p className="mt-1 text-sm text-gray-500">
                            {profile.achievements.find(a => !a.earned).description}
                          </p>
                          <div className="mt-2 w-full bg-gray-200 rounded-full h-2.5">
                            <div className="bg-yellow-400 h-2.5 rounded-full" style={{ width: '65%' }}></div>
                          </div>
                          <p className="mt-1 text-xs text-gray-500">65% complete</p>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-6">
                        <p className="text-gray-500">You've unlocked all available achievements! More coming soon.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'stats' && (
              <div className="card-surface overflow-hidden sm:rounded-lg">
                <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Game Statistics</h3>
                </div>
                <div className="px-4 py-5 sm:p-6">
                  <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-500">Games Played</h4>
                      <p className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.gamesPlayed}</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-500">Games Won</h4>
                      <p className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.gamesWon}</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-500">Win Rate</h4>
                      <p className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.winRate}%</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-500">Average Score</h4>
                      <p className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.averageScore}</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-500">Current Streak</h4>
                      <p className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.currentStreak} days</p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-500">Highest Streak</h4>
                      <p className="mt-1 text-3xl font-semibold text-gray-900">{profile.stats.highestStreak} days</p>
                    </div>
                  </div>
                  
                  <div className="mt-8">
                    <h4 className="text-sm font-medium text-gray-900 mb-4">Performance Over Time</h4>
                    <div className="bg-gray-100 p-4 rounded-lg h-64 flex items-center justify-center">
                      <p className="text-gray-500">Performance chart would be displayed here</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'achievements' && (
              <div className="card-surface overflow-hidden sm:rounded-lg">
                <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Achievements</h3>
                  <p className="mt-1 max-w-2xl text-sm text-gray-500">
                    {profile.achievements.filter(a => a.earned).length} of {profile.achievements.length} achievements unlocked
                  </p>
                </div>
                <div className="px-4 py-5 sm:p-6">
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    {profile.achievements.map((achievement) => (
                      <div 
                        key={achievement.id}
                        className={`relative rounded-lg border p-4 ${
                          achievement.earned 
                            ? 'border-transparent bg-gray-50' 
                            : 'border-gray-200 bg-white opacity-50'
                        }`}
                      >
                        <div className="flex items-start">
                          <div className={`flex-shrink-0 h-10 w-10 rounded-full flex items-center justify-center ${
                            achievement.earned ? 'bg-yellow-100' : 'bg-gray-200'
                          }`}>
                            <span className="text-xl">{achievement.icon}</span>
                          </div>
                          <div className="ml-4">
                            <h4 className="text-sm font-medium text-gray-900">
                              {achievement.name}
                              {achievement.earned && (
                                <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                  Unlocked
                                </span>
                              )}
                            </h4>
                            <p className="mt-1 text-sm text-gray-500">{achievement.description}</p>
                            {achievement.earned && achievement.date && (
                              <p className="mt-1 text-xs text-gray-400">
                                Unlocked on {new Date(achievement.date).toLocaleDateString()}
                              </p>
                            )}
                          </div>
                        </div>
                        {!achievement.earned && (
                          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75">
                            <ShieldCheckIcon className="h-6 w-6 text-gray-400" />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'history' && (
              <div className="card-surface overflow-hidden sm:rounded-lg">
                <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="text-lg leading-6 font-medium text-gray-900">Game History</h3>
                      <p className="mt-1 max-w-2xl text-sm text-gray-500">Your recent game sessions and results</p>
                    </div>
                    <div className="flex space-x-2">
                      <select
                        id="time-period"
                        name="time-period"
                        className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                        defaultValue="all"
                      >
                        <option value="week">Last 7 days</option>
                        <option value="month">Last 30 days</option>
                        <option value="all">All time</option>
                      </select>
                    </div>
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Game
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Date
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Score
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Position
                        </th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {profile.recentGames.map((game) => (
                        <tr key={game.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm font-medium text-gray-900">{game.name}</div>
                            <div className="text-sm text-gray-500">{game.id}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{new Date(game.date).toLocaleDateString()}</div>
                            <div className="text-sm text-gray-500">{new Date(game.date).toLocaleTimeString()}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">{game.score}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              game.position === 1 ? 'bg-yellow-100 text-yellow-800' :
                              game.position === 2 ? 'bg-gray-100 text-gray-800' :
                              game.position === 3 ? 'bg-amber-100 text-amber-800' :
                              'bg-blue-100 text-blue-800'
                            }`}>
                              #{game.position}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              game.status === 'won' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                            }`}>
                              {game.status === 'won' ? 'Won' : 'Lost'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="bg-gray-50 px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
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
                        <span className="font-medium">{profile.recentGames.length}</span> results
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

            {activeTab === 'leaderboard' && (
              <div className="card-surface overflow-hidden sm:rounded-lg">
                <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
                  <div className="flex justify-between items-center">
                    <div>
                      <h3 className="text-lg leading-6 font-medium text-gray-900">Global Leaderboard</h3>
                      <p className="mt-1 max-w-2xl text-sm text-gray-500">Top players by total score</p>
                    </div>
                    <div className="flex space-x-2">
                      <select
                        id="leaderboard-type"
                        name="leaderboard-type"
                        className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                        defaultValue="global"
                      >
                        <option value="global">Global</option>
                        <option value="friends">Friends</option>
                        <option value="weekly">This Week</option>
                        <option value="monthly">This Month</option>
                      </select>
                    </div>
                  </div>
                </div>
                <div className="bg-white overflow-hidden">
                  <ul className="divide-y divide-gray-200">
                    {leaderboard.map((player, index) => (
                      <li 
                        key={player.id}
                        className={`px-4 py-4 sm:px-6 ${player.isCurrentUser ? 'bg-indigo-50' : 'hover:bg-gray-50'}`}
                      >
                        <div className="flex items-center">
                          <div className="flex-shrink-0">
                            {index < 3 ? (
                              <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
                                index === 0 ? 'bg-yellow-100 text-yellow-800' :
                                index === 1 ? 'bg-gray-100 text-gray-800' :
                                'bg-amber-100 text-amber-800'
                              }`}>
                                <span className="font-medium">{index + 1}</span>
                              </div>
                            ) : (
                              <div className="h-8 w-8 rounded-full flex items-center justify-center bg-gray-50">
                                <span className="text-gray-500 font-medium">{index + 1}</span>
                              </div>
                            )}
                          </div>
                          <div className="ml-4 flex items-center flex-1 min-w-0">
                            <div className="flex-shrink-0 h-10 w-10">
                              <img 
                                className="h-10 w-10 rounded-full" 
                                src={player.avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(player.username)}&background=random`} 
                                alt="" 
                              />
                            </div>
                            <div className="ml-4 min-w-0 flex-1">
                              <div className="flex justify-between">
                                <p className={`text-sm font-medium ${
                                  player.isCurrentUser ? 'text-indigo-600' : 'text-gray-900'
                                } truncate`}>
                                  {player.username}
                                  {player.isCurrentUser && ' (You)'}
                                </p>
                                <div className="ml-2 flex-shrink-0 flex">
                                  <p className="text-sm text-gray-500">
                                    {player.score.toLocaleString()} pts
                                  </p>
                                </div>
                              </div>
                              <div className="mt-1 flex justify-between">
                                <p className="text-sm text-gray-500">
                                  {player.gamesPlayed} games • {player.winRate}% win rate
                                </p>
                                {player.isCurrentUser && (
                                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                                    Your Rank: #{index + 1}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                        {player.isCurrentUser && index < leaderboard.length - 1 && (
                          <div className="mt-4 pt-4 border-t border-gray-200">
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-500">Next rank (#{index}):</span>
                              <span className="font-medium">
                                {Math.ceil((leaderboard[index].score - player.score) / 1000) * 1000 - (leaderboard[index].score - player.score)} pts to go
                              </span>
                            </div>
                            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-indigo-600 h-2 rounded-full" 
                                style={{ 
                                  width: `${((leaderboard[index].score - player.score) / 1000) * 100}%` 
                                }}
                              ></div>
                            </div>
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-gray-50 px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
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
                        Showing <span className="font-medium">1</span> to <span className="font-medium">10</span> of{' '}
                        <span className="font-medium">{leaderboard.length}</span> results
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
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
