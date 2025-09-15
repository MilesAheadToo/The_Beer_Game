import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Toolbar,
  Box,
  Typography,
  Collapse,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  AccountTree as SupplyChainIcon,
  PlayCircleOutline as SimulationIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  SportsEsports as GamesIcon,
  AdminPanelSettings as AdminIcon,
  People as UsersIcon,
  Groups as GroupsIcon,
  ExpandLess,
  ExpandMore,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useAuth } from '../contexts/AuthContext';
import { isGroupAdmin as isGroupAdminUser, isSystemAdmin as isSystemAdminUser } from '../utils/authUtils';

const drawerWidth = 240;

const StyledDrawer = styled(Drawer)(({ theme }) => ({
  width: drawerWidth,
  flexShrink: 0,
  '& .MuiDrawer-paper': {
    width: drawerWidth,
    boxSizing: 'border-box',
    backgroundColor: theme.palette.background.paper,
    borderRight: `1px solid ${theme.palette.divider}`,
  },
}));

const menuItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Games', icon: <GamesIcon />, path: '/games' },
  { text: 'Supply Chain', icon: <SupplyChainIcon />, path: '/supply-chain' },
  { text: 'Simulation', icon: <SimulationIcon />, path: '/simulation' },
  { text: 'Analysis', icon: <AnalyticsIcon />, path: '/analysis' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];

const groupAdminMenuItems = [
  { text: 'Supply Chain Configs', icon: <SupplyChainIcon />, path: '/admin/group/supply-chain-configs' },
  { text: 'Group Management', icon: <GroupsIcon />, path: '/admin/groups' },
  { text: 'User Management', icon: <UsersIcon />, path: '/admin/users' },
];

const systemAdminMenuItems = [
  { text: 'System Config', icon: <SettingsIcon />, path: '/system-config' },
  { text: 'Group Management', icon: <GroupsIcon />, path: '/admin/groups' },
  { text: 'User Management', icon: <UsersIcon />, path: '/system/users' },
];

const Sidebar = ({ mobileOpen, handleDrawerToggle }) => {
  const [adminOpen, setAdminOpen] = React.useState(false);
  const { user: currentUser } = useAuth() || {};
  const isSystemAdmin = isSystemAdminUser(currentUser);
  const isGroupAdmin = isGroupAdminUser(currentUser);
  const location = useLocation();
  
  // If we're still loading the user, don't render anything
  if (!currentUser) {
    return null;
  }

  const drawer = (
    <div>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box component="img" src="/daybreak_logo.png" alt="Daybreak logo" sx={{ height: 28, width: 'auto', mr: 1 }} />
          <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
            Beer Game
          </Typography>
        </Box>
      </Toolbar>
      <Divider />
      <List>
        {!isSystemAdmin && menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'primary.light',
                  color: 'primary.contrastText',
                  '&:hover': {
                    backgroundColor: 'primary.light',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
              }}
            >
              <ListItemIcon sx={{ color: 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}

        {/* Admin Section for group administrators */}
        {!isSystemAdmin && isGroupAdmin && (
          <>
            <Divider sx={{ my: 1 }} />
            <ListItemButton onClick={() => setAdminOpen(!adminOpen)}>
              <ListItemIcon>
                <AdminIcon />
              </ListItemIcon>
              <ListItemText primary="Admin" />
              {adminOpen ? <ExpandLess /> : <ExpandMore />}
            </ListItemButton>
            <Collapse in={adminOpen} timeout="auto" unmountOnExit>
              <List component="div" disablePadding>
                {groupAdminMenuItems.map((item) => (
                  <ListItemButton
                    key={item.text}
                    component={Link}
                    to={item.path}
                    selected={location.pathname === item.path}
                    sx={{
                      pl: 4,
                      '&.Mui-selected': {
                        backgroundColor: 'primary.light',
                        color: 'primary.contrastText',
                        '&:hover': {
                          backgroundColor: 'primary.light',
                        },
                        '& .MuiListItemIcon-root': {
                          color: 'primary.contrastText',
                        },
                      },
                    }}
                  >
                    <ListItemIcon sx={{ color: 'inherit' }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText primary={item.text} />
                  </ListItemButton>
                ))}
              </List>
            </Collapse>
          </>
        )}

        {/* System administrator menu */}
        {isSystemAdmin && systemAdminMenuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              component={Link}
              to={item.path}
              selected={location.pathname === item.path}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: 'primary.light',
                  color: 'primary.contrastText',
                  '&:hover': {
                    backgroundColor: 'primary.light',
                  },
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
              }}
            >
              <ListItemIcon sx={{ color: 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </div>
  );

  return (
    <Box component="nav" sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}>
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better open performance on mobile.
        }}
        sx={{
          display: { xs: 'block', sm: 'none' },
          '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
        }}
      >
        {drawer}
      </Drawer>
      {/* Desktop drawer */}
      <StyledDrawer
        variant="permanent"
        sx={{
          display: { xs: 'none', sm: 'block' },
        }}
        open
      >
        {drawer}
      </StyledDrawer>
    </Box>
  );
};

export default Sidebar;
