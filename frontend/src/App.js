import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { Box, CssBaseline, ThemeProvider, CircularProgress } from '@mui/material';
import { theme } from './theme';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import SupplyChain from './pages/SupplyChain';
import Simulation from './pages/Simulation';
import Analysis from './pages/Analysis';
import Settings from './pages/Settings';
import GamesList from './pages/GamesList';
import MixedGamesList from './pages/MixedGamesList';
import CreateMixedGame from './pages/CreateMixedGame';
import GameBoard from './pages/GameBoard';
import Login from './pages/Login';
import UserManagement from './pages/admin/UserManagement';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { isAuthenticated, logout } from './services/authService';

const PrivateRoute = ({ children }) => {
  const [loading, setLoading] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);
  const location = useLocation();

  useEffect(() => {
    let isMounted = true;
    
    const checkAuth = async () => {
      try {
        const isAuth = await isAuthenticated();
        if (isMounted) {
          setAuthenticated(isAuth);
          setLoading(false);
        }
      } catch (error) {
        console.error('Authentication check failed:', error);
        if (isMounted) {
          setAuthenticated(false);
          setLoading(false);
        }
      }
    };

    checkAuth();
    
    return () => {
      isMounted = false;
    };
  }, [location]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!authenticated) {
    // Store the intended location to redirect back after login
    const from = location.pathname !== '/login' ? location.pathname + location.search : '/';
    return <Navigate to="/login" state={{ from }} replace />;
  }

  return children;
};

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const location = useLocation();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleLogout = () => {
    logout();
    setIsLoggedIn(false);
  };
  
  // Check if we're on a game page that needs WebSocket
  const isGamePage = location.pathname.startsWith('/games/');

  const renderContent = (children) => {
    if (isGamePage) {
      return <WebSocketProvider>{children}</WebSocketProvider>;
    }
    return children;
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Routes>
        <Route 
          path="/login" 
          element={
            isLoggedIn ? 
            <Navigate to="/" replace /> : 
            <Login onLogin={() => {
              setIsLoggedIn(true);
              // Force a page reload to ensure all auth state is properly initialized
              window.location.href = '/';
            }} />
          } 
        />
        <Route
          path="/*"
          element={
            <PrivateRoute>
              {renderContent(
                <Box sx={{ display: 'flex' }}>
                  <CssBaseline />
                  <Navbar handleDrawerToggle={handleDrawerToggle} onLogout={handleLogout} />
                  <Sidebar mobileOpen={mobileOpen} handleDrawerToggle={handleDrawerToggle} />
                  <Box
                    component="main"
                    sx={{
                      flexGrow: 1,
                      p: 3,
                      width: { sm: `calc(100% - ${240}px)` },
                      marginTop: '64px',
                    }}
                  >
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="games">
                        <Route index element={<GamesList />} />
                        <Route path="mixed" element={<MixedGamesList />} />
                        <Route path="mixed/new" element={<CreateMixedGame />} />
                        <Route path=":gameId" element={<GameBoard />} />
                        <Route path="mixed/:gameId" element={<GameBoard />} />
                      </Route>
                      <Route path="/supply-chain" element={<SupplyChain />} />
                      <Route path="/simulation" element={<Simulation />} />
                      <Route path="/analysis" element={<Analysis />} />
                      <Route path="/settings" element={
            <PrivateRoute>
              <Settings />
            </PrivateRoute>
          } />
          <Route path="/admin/users" element={
            <PrivateRoute>
              <UserManagement />
            </PrivateRoute>
          } />
                      <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                  </Box>
                </Box>
              )}
            </PrivateRoute>
          }
        />
        {/* Redirect root to login if not authenticated */}
        <Route 
          path="/" 
          element={
            isLoggedIn ? 
            <Navigate to="/dashboard" replace /> : 
            <Navigate to="/login" replace />
          } 
        />
      </Routes>
    </ThemeProvider>
  );
}

export default App;
