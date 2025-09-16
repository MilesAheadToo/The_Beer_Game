import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Box,
  CircularProgress,
  Paper,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import StorageIcon from '@mui/icons-material/Storage';
import SportsEsportsIcon from '@mui/icons-material/SportsEsports';
import GroupIcon from '@mui/icons-material/Group';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { Navigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { isSystemAdmin as isSystemAdminUser } from '../../utils/authUtils';
import GroupSupplyChainConfigList from './GroupSupplyChainConfigList';
import GroupPlayerManagement from './UserManagement';
import GroupGameConfigPanel from './GroupGameConfigPanel';
import GroupGameSupervisionPanel from './GroupGameSupervisionPanel';
import { mixedGameApi } from '../../services/api';

const tabItems = [
  { value: 'sc', label: 'SC Config', icon: <StorageIcon fontSize="small" /> },
  { value: 'game', label: 'Game Config', icon: <SportsEsportsIcon fontSize="small" /> },
  { value: 'users', label: 'User Config', icon: <GroupIcon fontSize="small" /> },
  { value: 'supervision', label: 'Game Supervision', icon: <VisibilityIcon fontSize="small" /> },
];

const AdminDashboard = () => {
  const { user, loading, isGroupAdmin } = useAuth();
  const isSystemAdmin = isSystemAdminUser(user);
  const [searchParams, setSearchParams] = useSearchParams();

  const initialTab = searchParams.get('section') || 'sc';
  const [activeTab, setActiveTab] = useState(tabItems.some((tab) => tab.value === initialTab) ? initialTab : 'sc');

  const handleTabChange = (_event, newValue) => {
    setActiveTab(newValue);
    setSearchParams({ section: newValue });
  };

  const [games, setGames] = useState([]);
  const [gamesLoading, setGamesLoading] = useState(true);
  const [gamesError, setGamesError] = useState(null);

  const refreshGames = useCallback(async () => {
    setGamesLoading(true);
    try {
      const list = await mixedGameApi.getGames();
      setGames(Array.isArray(list) ? list : []);
      setGamesError(null);
    } catch (error) {
      const detail = error?.response?.data?.detail || error?.message || 'Unable to load games right now.';
      setGames([]);
      setGamesError(detail);
    } finally {
      setGamesLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshGames();
  }, [refreshGames]);

  const groupId = useMemo(() => {
    const rawGroup = user?.group_id;
    if (rawGroup == null) return null;
    const parsed = Number(rawGroup);
    return Number.isFinite(parsed) ? parsed : null;
  }, [user]);

  const currentUserId = user?.id ?? null;

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!user) {
    return <Navigate to="/unauthorized" replace />;
  }

  if (isSystemAdmin && !isGroupAdmin) {
    return <Navigate to="/admin/groups" replace />;
  }

  if (!isGroupAdmin) {
    return <Navigate to="/unauthorized" replace />;
  }

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', py: 4, px: { xs: 2, md: 3 } }}>
      <Paper elevation={0} sx={{ mb: 4, p: { xs: 2, md: 3 } }}>
        <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
          Group Administrator Workspace
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Configure your supply chain templates, manage player access, and supervise active games from a single workspace.
        </Typography>
        <Tabs
          value={activeTab}
          onChange={handleTabChange}
          variant="scrollable"
          allowScrollButtonsMobile
          sx={{ mt: 3 }}
        >
          {tabItems.map((tab) => (
            <Tab
              key={tab.value}
              value={tab.value}
              iconPosition="start"
              icon={tab.icon}
              label={tab.label}
              sx={{ textTransform: 'none', fontWeight: 600 }}
            />
          ))}
        </Tabs>
      </Paper>

      <Box>
        {activeTab === 'sc' && (
          <GroupSupplyChainConfigList />
        )}

        {activeTab === 'game' && (
          <GroupGameConfigPanel
            games={games}
            loading={gamesLoading}
            error={gamesError}
            onRefresh={refreshGames}
            groupId={groupId}
            currentUserId={currentUserId}
          />
        )}

        {activeTab === 'users' && <GroupPlayerManagement />}

        {activeTab === 'supervision' && (
          <GroupGameSupervisionPanel
            games={games}
            loading={gamesLoading}
            error={gamesError}
            onRefresh={refreshGames}
            groupId={groupId}
            currentUserId={currentUserId}
          />
        )}
      </Box>
    </Box>
  );
};

export default AdminDashboard;
