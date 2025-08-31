import React from 'react';
import { AppBar, Toolbar, IconButton, Typography, Box, Avatar } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SettingsIcon from '@mui/icons-material/Settings';
import { styled } from '@mui/material/styles';

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  zIndex: theme.zIndex.drawer + 1,
  transition: theme.transitions.create(['width', 'margin'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
}));

const Navbar = ({ handleDrawerToggle }) => {
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
        <Box sx={{ flexGrow: 1 }} />
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton color="inherit">
            <NotificationsIcon />
          </IconButton>
          <IconButton color="inherit">
            <SettingsIcon />
          </IconButton>
          <IconButton color="inherit">
            <Avatar sx={{ width: 32, height: 32 }}>U</Avatar>
          </IconButton>
        </Box>
      </Toolbar>
    </StyledAppBar>
  );
};

export default Navbar;
