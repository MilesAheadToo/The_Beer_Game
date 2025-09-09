import { useState, useEffect } from 'react';
import { Typography } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { useHelp } from '../contexts/HelpContext';
import { 
  UserIcon, 
  BellIcon, 
  Cog6ToothIcon, 
  ShieldCheckIcon, 
  MoonIcon, 
  SunIcon, 
  ComputerDesktopIcon,
  CheckIcon
} from '@heroicons/react/24/outline';

const Settings = () => {
  const { user, updateProfile } = useAuth();
  const { openHelp } = useHelp();
  
  // Settings state
  const [settings, setSettings] = useState({
    theme: 'system',
    notifications: {
      email: true,
      inApp: true,
      sound: true,
    },
    privacy: {
      showOnlineStatus: true,
      allowFriendRequests: true,
      showInLeaderboards: true,
    },
    game: {
      animationSpeed: 'normal',
      confirmBeforeLeavingGame: true,
      showTutorialTips: true,
    },
  });
  
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState({ type: '', message: '' });
  
  // Load saved settings when component mounts
  useEffect(() => {
    const savedSettings = localStorage.getItem('gameSettings');
    if (savedSettings) {
      try {
        setSettings(JSON.parse(savedSettings));
      } catch (error) {
        console.error('Failed to parse saved settings', error);
      }
    }
  }, []);
  
  // Save settings to localStorage and update UI state
  const saveSettings = async (newSettings) => {
    setIsSaving(true);
    setSaveStatus({ type: '', message: '' });
    
    try {
      // In a real app, you would save these to your backend
      // await api.updateUserSettings(user.id, newSettings);
      
      // For now, just save to localStorage
      localStorage.setItem('gameSettings', JSON.stringify(newSettings));
      
      // Update local state
      setSettings(newSettings);
      
      // Apply theme if changed
      if (settings.theme !== newSettings.theme) {
        applyTheme(newSettings.theme);
      }
      
      setSaveStatus({ 
        type: 'success', 
        message: 'Settings saved successfully!' 
      });
    } catch (error) {
      console.error('Failed to save settings', error);
      setSaveStatus({ 
        type: 'error', 
        message: 'Failed to save settings. Please try again.' 
      });
    } finally {
      setIsSaving(false);
      
      // Clear success message after 3 seconds
      if (saveStatus.type === 'success') {
        setTimeout(() => {
          setSaveStatus({ type: '', message: '' });
        }, 3000);
      }
    }
  };
  
  // Apply theme to document
  const applyTheme = (theme) => {
    const root = window.document.documentElement;
    
    // Remove all theme classes
    root.classList.remove('light', 'dark');
    
    if (theme === 'system') {
      // Use system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.add(prefersDark ? 'dark' : 'light');
    } else {
      root.classList.add(theme);
    }
    
    // Add a class to the body to indicate the theme
    document.body.className = `theme-${theme}`;
  };
  
  // Handle setting changes
  const handleSettingChange = (section, key, value) => {
    const newSettings = {
      ...settings,
      [section]: {
        ...settings[section],
        [key]: value
      }
    };
    
    // Special handling for theme changes
    if (section === 'theme') {
      newSettings.theme = value;
      applyTheme(value);
    }
    
    saveSettings(newSettings);
  };
  
  // Toggle boolean settings
  const toggleSetting = (section, key) => {
    handleSettingChange(section, key, !settings[section][key]);
  };
  
  // Render a section header
  const SectionHeader = ({ icon: Icon, title, description }) => (
    <div className="md:col-span-1">
      <div className="pad-6 sm:px-0">
        <div className="flex items-center">
          <Icon className="h-5 w-5 mr-2 text-gray-500" />
          <Typography variant="h5" component="h3">{title}</Typography>
        </div>
        <Typography variant="subtitle2" sx={{ mt: 1 }}>{description}</Typography>
      </div>
    </div>
  );
  
  // Render a setting control
  const SettingControl = ({ label, description, children, className = '' }) => (
    <div className={`pad-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex-grow">
          <Typography variant="body1" sx={{ fontWeight: 600 }}>{label}</Typography>
          {description && (<Typography variant="body2" color="text.secondary" sx={{ mt: .5 }}>{description}</Typography>)}
        </div>
        <div className="ml-4">
          {children}
        </div>
      </div>
    </div>
  );
  
  // Render a toggle switch
  const ToggleSwitch = ({ checked, onChange, id }) => (
    <button
      type="button"
      className={`${checked ? 'bg-indigo-600' : 'bg-gray-200'} relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      id={id}
    >
      <span className="sr-only">Toggle {id}</span>
      <span
        aria-hidden="true"
        className={`${checked ? 'translate-x-5' : 'translate-x-0'} pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out`}
      />
    </button>
  );
  
  // Render a select dropdown
  const Select = ({ value, onChange, options, id, className = '' }) => (
    <select
      id={id}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md ${className}`}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );

  return (
    <div className="max-w-7xl mx-auto pad-8">
      <div className="md:flex md:items-center md:justify-between mb-8">
        <div className="flex-1 min-w-0">
          <Typography variant="h3" component="h2">Settings</Typography>
          <Typography variant="subtitle2" sx={{ mt: .5 }}>Manage your account settings and preferences</Typography>
        </div>
        <div className="mt-4 flex md:mt-0 md:ml-4
        ">
          <button
            type="button"
            onClick={() => openHelp('settings')}
            className="ml-3 inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-indigo-700 bg-indigo-100 hover:bg-indigo-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <QuestionMarkCircleIcon className="-ml-1 mr-2 h-5 w-5" />
            Help
          </button>
        </div>
      </div>
      
      {saveStatus.message && (
        <div className={`mb-6 rounded-md p-4 ${saveStatus.type === 'error' ? 'bg-red-50' : 'bg-green-50'}`}>
          <div className="flex">
            <div className="flex-shrink-0">
              {saveStatus.type === 'error' ? (
                <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              ) : (
                <CheckIcon className="h-5 w-5 text-green-400" aria-hidden="true" />
              )}
            </div>
            <div className="ml-3">
              <p className={`text-sm font-medium ${saveStatus.type === 'error' ? 'text-red-800' : 'text-green-800'}`}>
                {saveStatus.message}
              </p>
            </div>
          </div>
        </div>
      )}
      
      <div className="mt-10 divide-y divide-gray-200">
        {/* Profile Settings */}
        <div className="space-y-6 py-6">
          <div className="md:grid md:grid-cols-3 md:gap-6">
            <SectionHeader 
              icon={UserIcon} 
              title="Profile" 
              description="Update your profile information and avatar" 
            />
            <div className="mt-5 md:mt-0 md:col-span-2">
              <div className="card-surface overflow-hidden sm:rounded-lg">
                <div className="pad-6">
                  <div className="grid grid-cols-6 gap-6">
                    <div className="col-span-6 sm:col-span-4">
                      <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                        Username
                      </label>
                      <input
                        type="text"
                        name="username"
                        id="username"
                        value={user?.username || ''}
                        disabled={isSaving}
                        className="mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                      />
                    </div>
                    
                    <div className="col-span-6 sm:col-span-4">
                      <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                        Email address
                      </label>
                      <input
                        type="email"
                        name="email"
                        id="email"
                        value={user?.email || ''}
                        disabled={isSaving}
                        className="mt-1 focus:ring-indigo-500 focus:border-indigo-500 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                      />
                    </div>
                    
                    <div className="col-span-6">
                      <label htmlFor="bio" className="block text-sm font-medium text-gray-700">
                        Bio
                      </label>
                      <div className="mt-1">
                        <textarea
                          id="bio"
                          name="bio"
                          rows={3}
                          className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 mt-1 block w-full sm:text-sm border border-gray-300 rounded-md"
                          placeholder="Tell us a little about yourself"
                          defaultValue={''}
                        />
                      </div>
                      <p className="mt-2 text-sm text-gray-500">
                        Brief description for your profile. URLs are hyperlinked.
                      </p>
                    </div>
                    
                    <div className="col-span-6">
                      <label className="block text-sm font-medium text-gray-700">
                        Photo
                      </label>
                      <div className="mt-1 flex items-center">
                        <span className="h-12 w-12 rounded-full overflow-hidden bg-gray-100">
                          <svg className="h-full w-full text-gray-300" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
                          </svg>
                        </span>
                        <button
                          type="button"
                          className="ml-5 bg-white py-2 px-3 border border-gray-300 rounded-md shadow-sm text-sm leading-4 font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                        >
                          Change
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="pad-6 text-right">
                  <button
                    type="submit"
                    className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    disabled={isSaving}
                  >
                    {isSaving ? 'Saving...' : 'Save'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Theme Settings */}
        <div className="py-6">
          <div className="md:grid md:grid-cols-3 md:gap-6">
            <SectionHeader 
              icon={settings.theme === 'dark' ? MoonIcon : settings.theme === 'light' ? SunIcon : ComputerDesktopIcon} 
              title="Appearance" 
              description="Customize how the app looks and feels" 
            />
            <div className="mt-5 md:mt-0 md:col-span-2">
              <div className="card-surface overflow-hidden sm:rounded-lg">
                <div className="pad-6 space-y-6">
                  <SettingControl
                    label="Theme"
                    description="Select your preferred theme"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center">
                        <input
                          id="theme-light"
                          name="theme"
                          type="radio"
                          checked={settings.theme === 'light'}
                          onChange={() => handleSettingChange('theme', 'theme', 'light')}
                          className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300"
                        />
                        <label htmlFor="theme-light" className="ml-2 block text-sm text-gray-700">
                          <div className="flex items-center">
                            <SunIcon className="h-5 w-5 mr-1 text-yellow-500" />
                            Light
                          </div>
                        </label>
                      </div>
                      
                      <div className="flex items-center">
                        <input
                          id="theme-dark"
                          name="theme"
                          type="radio"
                          checked={settings.theme === 'dark'}
                          onChange={() => handleSettingChange('theme', 'theme', 'dark')}
                          className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300"
                        />
                        <label htmlFor="theme-dark" className="ml-2 block text-sm text-gray-700">
                          <div className="flex items-center">
                            <MoonIcon className="h-5 w-5 mr-1 text-indigo-500" />
                            Dark
                          </div>
                        </label>
                      </div>
                      
                      <div className="flex items-center">
                        <input
                          id="theme-system"
                          name="theme"
                          type="radio"
                          checked={settings.theme === 'system'}
                          onChange={() => handleSettingChange('theme', 'theme', 'system')}
                          className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300"
                        />
                        <label htmlFor="theme-system" className="ml-2 block text-sm text-gray-700">
                          <div className="flex items-center">
                            <ComputerDesktopIcon className="h-5 w-5 mr-1 text-gray-500" />
                            System
                          </div>
                        </label>
                      </div>
                    </div>
                  </SettingControl>
                  
                  <SettingControl
                    label="Animation Speed"
                    description="Adjust the speed of animations in the app"
                  >
                    <Select
                      id="animation-speed"
                      value={settings.game.animationSpeed}
                      onChange={(value) => handleSettingChange('game', 'animationSpeed', value)}
                      options={[
                        { value: 'fast', label: 'Fast' },
                        { value: 'normal', label: 'Normal' },
                        { value: 'slow', label: 'Slow' },
                        { value: 'off', label: 'Off' },
                      ]}
                      className="w-32"
                    />
                  </SettingControl>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Notification Settings */}
        <div className="py-6">
          <div className="md:grid md:grid-cols-3 md:gap-6">
            <SectionHeader 
              icon={BellIcon} 
              title="Notifications" 
              description="Manage how you receive notifications" 
            />
            <div className="mt-5 md:mt-0 md:col-span-2">
              <div className="card-surface overflow-hidden sm:rounded-lg">
                <div className="pad-6 space-y-6">
                  <SettingControl
                    label="Email notifications"
                    description="Receive email notifications"
                  >
                    <ToggleSwitch 
                      checked={settings.notifications.email} 
                      onChange={() => toggleSetting('notifications', 'email')} 
                      id="email-notifications"
                    />
                  </SettingControl>
                  
                  <SettingControl
                    label="In-app notifications"
                    description="Show notifications within the app"
                  >
                    <ToggleSwitch 
                      checked={settings.notifications.inApp} 
                      onChange={() => toggleSetting('notifications', 'inApp')} 
                      id="in-app-notifications"
                    />
                  </SettingControl>
                  
                  <SettingControl
                    label="Sound effects"
                    description="Play sound for notifications"
                    className="border-t border-gray-200"
                  >
                    <ToggleSwitch 
                      checked={settings.notifications.sound} 
                      onChange={() => toggleSetting('notifications', 'sound')} 
                      id="sound-effects"
                    />
                  </SettingControl>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Privacy Settings */}
        <div className="py-6">
          <div className="md:grid md:grid-cols-3 md:gap-6">
            <SectionHeader 
              icon={ShieldCheckIcon} 
              title="Privacy" 
              description="Control your privacy settings" 
            />
            <div className="mt-5 md:mt-0 md:col-span-2">
              <div className="shadow overflow-hidden sm:rounded-md">
                <div className="px-4 py-5 bg-white space-y-6 sm:p-6">
                  <SettingControl
                    label="Show online status"
                    description="Allow others to see when you're online"
                  >
                    <ToggleSwitch 
                      checked={settings.privacy.showOnlineStatus} 
                      onChange={() => toggleSetting('privacy', 'showOnlineStatus')} 
                      id="online-status"
                    />
                  </SettingControl>
                  
                  <SettingControl
                    label="Allow friend requests"
                    description="Let other players send you friend requests"
                  >
                    <ToggleSwitch 
                      checked={settings.privacy.allowFriendRequests} 
                      onChange={() => toggleSetting('privacy', 'allowFriendRequests')} 
                      id="friend-requests"
                    />
                  </SettingControl>
                  
                  <SettingControl
                    label="Show in leaderboards"
                    description="Include your stats in public leaderboards"
                    className="border-t border-gray-200"
                  >
                    <ToggleSwitch 
                      checked={settings.privacy.showInLeaderboards} 
                      onChange={() => toggleSetting('privacy', 'showInLeaderboards')} 
                      id="leaderboards"
                    />
                  </SettingControl>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Game Settings */}
        <div className="py-6">
          <div className="md:grid md:grid-cols-3 md:gap-6">
            <SectionHeader 
              icon={Cog6ToothIcon} 
              title="Game Settings" 
              description="Customize your game experience" 
            />
            <div className="mt-5 md:mt-0 md:col-span-2">
              <div className="shadow overflow-hidden sm:rounded-md">
                <div className="px-4 py-5 bg-white space-y-6 sm:p-6">
                  <SettingControl
                    label="Confirm before leaving game"
                    description="Show a confirmation dialog when leaving a game in progress"
                  >
                    <ToggleSwitch 
                      checked={settings.game.confirmBeforeLeavingGame} 
                      onChange={() => toggleSetting('game', 'confirmBeforeLeavingGame')} 
                      id="confirm-leave"
                    />
                  </SettingControl>
                  
                  <SettingControl
                    label="Show tutorial tips"
                    description="Display helpful tips and tutorials"
                  >
                    <ToggleSwitch 
                      checked={settings.game.showTutorialTips} 
                      onChange={() => toggleSetting('game', 'showTutorialTips')} 
                      id="tutorial-tips"
                    />
                  </SettingControl>
                </div>
                
                <div className="pad-6 text-right">
                  <button
                    type="button"
                    onClick={() => {
                      // Reset to default settings
                      const defaultSettings = {
                        theme: 'system',
                        notifications: {
                          email: true,
                          inApp: true,
                          sound: true,
                        },
                        privacy: {
                          showOnlineStatus: true,
                          allowFriendRequests: true,
                          showInLeaderboards: true,
                        },
                        game: {
                          animationSpeed: 'normal',
                          confirmBeforeLeavingGame: true,
                          showTutorialTips: true,
                        },
                      };
                      saveSettings(defaultSettings);
                    }}
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Reset to Defaults
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
