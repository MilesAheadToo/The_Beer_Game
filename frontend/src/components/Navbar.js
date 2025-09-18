import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, 
  Toolbar, 
  IconButton, 
  Typography, 
  Box, 
  Avatar, 
  Menu, 
  MenuItem, 
  Divider, 
  ListItemIcon, 
  Button,
  Tooltip
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SettingsIcon from '@mui/icons-material/Settings';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SportsEsportsIcon from '@mui/icons-material/SportsEsports';
import GroupIcon from '@mui/icons-material/Group';
import GroupsIcon from '@mui/icons-material/Groups';
import Logout from '@mui/icons-material/Logout';
import { styled } from '@mui/material/styles';
import { useAuth } from '../contexts/AuthContext';
import { isGroupAdmin as isGroupAdminUser, isSystemAdmin as isSystemAdminUser } from '../utils/authUtils';

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  zIndex: theme.zIndex.drawer + 1,
  transition: theme.transitions.create(['width', 'margin'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
}));

const Navbar = ({ handleDrawerToggle }) => {
  const { logout, user, isGroupAdmin } = useAuth();
  const [anchorEl, setAnchorEl] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const open = Boolean(anchorEl);

  const systemAdmin = isSystemAdminUser(user);
  const groupAdmin = isGroupAdmin || isGroupAdminUser(user);
  
  const isActive = (path) => {
    return location.pathname === path || 
           (path !== '/' && location.pathname.startsWith(path));
  };

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      await logout();
      handleClose();
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <StyledAppBar position="fixed" sx={{ width: '100%' }}>
      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="open drawer"
          edge="start"
          onClick={handleDrawerToggle}
          sx={{ mr: 2, display: { sm: 'none' } }}
        >
          <MenuIcon />
        </IconButton>
        <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
          <Box component="img" src="/daybreak_logo.png" alt="Daybreak logo" sx={{ height: 28, width: 'auto', mr: 1 }} />
          <Typography variant="h6" noWrap component="div">
            The Beer Game
          </Typography>
        </Box>
        <Box sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' }, ml: 4 }}>
          {systemAdmin ? null : groupAdmin ? (
            <>
              <Button
                color="inherit"
                startIcon={<DashboardIcon />}
                onClick={() => navigate('/admin')}
                sx={{
                  mx: 1,
                  bgcolor: isActive('/admin') ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.15)'
                  }
                }}
              >
                Admin Dashboard
              </Button>
              <Button
                color="inherit"
                startIcon={<SportsEsportsIcon />}
                onClick={() => navigate('/games')}
                sx={{
                  mx: 1,
                  bgcolor: isActive('/games') ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.15)'
                  }
                }}
              >
                Manage Games
              </Button>
              <Button
                color="inherit"
                startIcon={<GroupsIcon />}
                onClick={() => navigate('/admin/groups')}
                sx={{
                  mx: 1,
                  bgcolor: isActive('/admin/groups') ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.15)'
                  }
                }}
              >
                Group Management
              </Button>
              <Button
                color="inherit"
                startIcon={<GroupIcon />}
                onClick={() => navigate('/admin/users')}
                sx={{
                  mx: 1,
                  bgcolor: isActive('/admin/users') ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.15)'
                  }
                }}
              >
                Manage Users
              </Button>
            </>
          ) : (
            <>
              <Button
                color="inherit"
                startIcon={<DashboardIcon />}
                onClick={() => navigate('/dashboard')}
                sx={{
                  mx: 1,
                  bgcolor: isActive('/dashboard') ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.15)'
                  }
                }}
              >
                My Dashboard
              </Button>
              <Button
                color="inherit"
                startIcon={<SportsEsportsIcon />}
                onClick={() => navigate('/games')}
                sx={{
                  mx: 1,
                  bgcolor: isActive('/games') ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.15)'
                  }
                }}
              >
                Games
              </Button>
            </>
          )}
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tooltip title="Notifications">
            <IconButton color="inherit">
              <NotificationsIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Settings">
            <IconButton color="inherit" onClick={() => navigate('/settings')}>
              <SettingsIcon />
            </IconButton>
          </Tooltip>
          <IconButton
            onClick={handleClick}
            size="small"
            sx={{ ml: 2 }}
            aria-controls={open ? 'account-menu' : undefined}
            aria-haspopup="true"
            aria-expanded={open ? 'true' : undefined}
          >
            <Avatar 
              sx={{ 
                width: 32, 
                height: 32,
                bgcolor: 'primary.light',
                color: 'primary.contrastText'
              }}
            >
              {user?.username?.charAt(0).toUpperCase() || 'U'}
            </Avatar>
          </IconButton>
        </Box>
      </Toolbar>
      <Menu
        anchorEl={anchorEl}
        id="account-menu"
        open={open}
        onClose={handleClose}
        onClick={handleClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: 'visible',
            filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
            mt: 1.5,
            '& .MuiAvatar-root': {
              width: 32,
              height: 32,
              ml: -0.5,
              mr: 1,
            },
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
        <MenuItem onClick={() => navigate('/settings')}>
          <Avatar /> Profile
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => navigate('/profile')}>
          <ListItemIcon>
            <Avatar sx={{ width: 24, height: 24, fontSize: '0.8rem' }} />
          </ListItemIcon>
          Profile
        </MenuItem>
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <Logout fontSize="small" />
          </ListItemIcon>
          Logout
        </MenuItem>
        {(groupAdmin || systemAdmin) && (
          <Box sx={{ p: 2, pt: 1, fontSize: '0.75rem', color: 'text.secondary' }}>
            <Box>{systemAdmin ? 'System Administrator' : 'Group Administrator'}</Box>
            <Box>{user.email}</Box>
          </Box>
        )}
      </Menu>
    </StyledAppBar>
  );
};

export default Navbar;
