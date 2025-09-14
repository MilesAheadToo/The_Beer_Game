import { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation, useParams } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Button,
  Box,
  Menu,
  MenuItem,
  Divider,
  ListItemIcon,
  ListItemText,
  Avatar,
  Badge,
  Tooltip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  Person as PersonIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  Dashboard as DashboardIcon,
  SportsEsports as GamesIcon,
  PersonOutline as PlayersIcon,
  HelpOutline as HelpIcon,
  NotificationsNone as NotificationsIcon
} from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import mixedGameApi from '../services/api';

const Navbar = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const [currentPath, setCurrentPath] = useState('');
  const [anchorEl, setAnchorEl] = useState(null);
  const [gameInfo, setGameInfo] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const { gameId } = useParams();
  const open = Boolean(anchorEl);
  
  // Update current path when location changes
  useEffect(() => {
    setCurrentPath(location.pathname);
  }, [location]);

  // Load game information when on a game page
  useEffect(() => {
    const fetchGameInfo = async () => {
      if (gameId) {
        try {
          const data = await mixedGameApi.getGame(gameId);
          setGameInfo(data);
        } catch (err) {
          console.error('Failed to load game info', err);
        }
      } else {
        setGameInfo(null);
      }
    };
    fetchGameInfo();
  }, [gameId]);

  const navigation = [
    { name: 'Dashboard', path: '/dashboard', icon: <DashboardIcon />, auth: true },
    { name: 'Games', path: '/games', icon: <GamesIcon />, auth: true },
    { name: 'Players', path: '/players', icon: <PlayersIcon />, auth: true },
  ];

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      await logout();
      handleMenuClose();
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const getInitials = (name) => {
    if (!name) return '';
    return name
      .split(' ')
      .map(part => part[0])
      .join('')
      .toUpperCase()
      .substring(0, 2);
  };

  const groupName = user?.group?.name || gameInfo?.group?.name;
  const scName = gameInfo?.config?.name;
  const gameName = gameInfo?.name;

  if (!isAuthenticated) {
    return null; // Don't show navbar for unauthenticated users
  }

  return (
    <AppBar 
      position="fixed"
      sx={{
        background: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)',
        boxShadow: '0 2px 20px rgba(0, 0, 0, 0.1)',
        borderBottom: '1px solid rgba(0, 0, 0, 0.05)',
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between', px: { xs: 2, md: 4 } }}>
        {/* Left side - Logo */}
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography
            variant="h6"
            component={Link}
            to="/dashboard"
            sx={{
              fontWeight: 700,
              color: 'primary.main',
              textDecoration: 'none',
              display: 'flex',
              alignItems: 'center',
              mr: 4,
            }}
          >
            The Beer Game
          </Typography>
          {(groupName || scName || gameName) && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ ml: 2 }}
            >
              {[groupName, scName, gameName].filter(Boolean).join(' | ')}
            </Typography>
          )}

          {/* Navigation Links - Desktop */}
          <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 1 }}>
            {navigation.map((item) => (
              <Button
                key={item.name}
                component={Link}
                to={item.path}
                startIcon={item.icon}
                sx={{
                  color: currentPath === item.path ? 'primary.main' : 'text.secondary',
                  fontWeight: currentPath === item.path ? 600 : 400,
                  '&:hover': {
                    backgroundColor: 'rgba(0, 0, 0, 0.02)',
                  },
                }}
              >
                {item.name}
              </Button>
            ))}
          </Box>
        </Box>

        {/* Right side - User menu */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tooltip title="Help">
            <IconButton color="inherit" onClick={() => navigate('/help')}>
              <HelpIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Notifications">
            <IconButton color="inherit">
              <Badge badgeContent={3} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>

          {/* User Menu */}
          <Box sx={{ display: 'flex', alignItems: 'center', ml: 1 }}>
            <Button
              onClick={handleMenuOpen}
              startIcon={
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    bgcolor: 'primary.main',
                    color: 'white',
                    fontSize: '0.875rem',
                  }}
                >
                  {getInitials(user?.name || '')}
                </Avatar>
              }
              endIcon={open ? <CloseIcon /> : <MenuIcon />}
              sx={{
                color: 'text.primary',
                textTransform: 'none',
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.02)',
                },
              }}
            >
              <Box sx={{ display: { xs: 'none', sm: 'block' }, ml: 1 }}>
                <Typography variant="body2" fontWeight={500}>
                  {user?.name || 'User'}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {user?.role || 'Player'}
                </Typography>
              </Box>
            </Button>

            {/* User Menu Dropdown */}
            <Menu
              anchorEl={anchorEl}
              open={open}
              onClose={handleMenuClose}
              onClick={handleMenuClose}
              PaperProps={{
                elevation: 0,
                sx: {
                  overflow: 'visible',
                  filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.1))',
                  mt: 1.5,
                  '&:before': {
                    content: '""',
                    display: 'block',
                    position: 'absolute',
                    top: 0,
                    right: 14,
                    width: 10,
                    height: 10,
                    bgcolor: 'background.paper',
                    transform: 'translateY(-50%) rotate(45deg)',
                    zIndex: 0,
                  },
                },
              }}
              transformOrigin={{ horizontal: 'right', vertical: 'top' }}
              anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            >
              <MenuItem onClick={() => navigate('/profile')}>
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText>Profile</ListItemText>
              </MenuItem>
              <MenuItem onClick={() => navigate('/settings')}>
                <ListItemIcon>
                  <SettingsIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText>Settings</ListItemText>
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" color="error" />
                </ListItemIcon>
                <ListItemText primaryTypographyProps={{ color: 'error' }}>
                  Logout
                </ListItemText>
              </MenuItem>
            </Menu>
          </Box>
        </Box>
      </Toolbar>

      {/* Mobile Navigation */}
      <Box sx={{ display: { xs: 'flex', md: 'none' }, borderTop: '1px solid', borderColor: 'divider' }}>
        {navigation.map((item) => (
          <Button
            key={item.name}
            component={Link}
            to={item.path}
            fullWidth
            sx={{
              py: 1.5,
              color: currentPath === item.path ? 'primary.main' : 'text.secondary',
              borderBottom: currentPath === item.path ? '2px solid' : 'none',
              borderColor: 'primary.main',
              borderRadius: 0,
            }}
          >
            {item.icon}
          </Button>
        ))}
      </Box>
    </AppBar>
  );
};

export default Navbar;
