import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
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
import './utils/fetchInterceptor';

// Global error handler
window.onerror = function(message, source, lineno, colno, error) {
  console.error('Global error:', { message, source, lineno, colno, error });
  return false; // Don't suppress default error handling
};

// Catch unhandled promise rejections
window.onunhandledrejection = function(event) {
  console.error('Unhandled rejection (promise):', event.reason);
};

const PrivateRoute = ({ children }) => {
  const location = useLocation();
  const token = localStorage.getItem('access_token');
  const isAuthed = !!token;
  
  console.log('PrivateRoute - access_token exists:', isAuthed);
  console.log('Current path:', location.pathname);

  if (!isAuthed) {
    const from = location.pathname !== '/login' ? location.pathname + location.search : '/';
    console.log('Not authenticated, redirecting to login. Will return to:', from);
    return <Navigate to="/login" state={{ from: location }} replace />;

  }

  console.log('User is authenticated, rendering protected content');
  return children;
};

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  // Check authentication status on component mount and when location changes
  useEffect(() => {
    console.log('=== Auth Check ===');
    console.log('Current path:', location.pathname);
    
    const checkAuth = () => {
      try {
        // Check for token in localStorage
        const token = localStorage.getItem('access_token');
        const isAuth = !!token;
        
        console.log('Auth check - access_token exists:', isAuth);
        
        // Update the logged in state if it doesn't match
        if (isAuth !== isLoggedIn) {
          console.log('Updating isLoggedIn state to:', isAuth);
          setIsLoggedIn(isAuth);
        }
        
        // If not authenticated and not on login page, redirect to login
        if (!isAuth && !location.pathname.startsWith('/login')) {
          console.log('Not authenticated, redirecting to login');
          navigate('/login', { state: { from: location }, replace: true });
        }
      } catch (error) {
        console.error('Authentication check failed:', error);
        setIsLoggedIn(false);
        if (!location.pathname.startsWith('/login')) {
          navigate('/login', { state: { from: location }, replace: true });
        }
      }
    };
    
    checkAuth();
  }, [location, navigate, isLoggedIn]);

  // Check if current route is the login page
  const isLoginPage = location.pathname === '/login';

  const handleLogout = async () => {
    try {
      await logout();
      setIsLoggedIn(false);
      // Clear any stored tokens
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
      // Redirect to login page
      navigate('/login', { replace: true });
    } catch (error) {
      console.error('Error during logout:', error);
    }
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
            <Navigate to="/dashboard" replace /> : 
            <Login onLogin={() => {
              console.log('onLogin callback called');
              // Update the login state
              setIsLoggedIn(true);
              // Redirect to dashboard or the originally requested page
              const from = location.state?.from?.pathname || '/dashboard';
              console.log('Redirecting to:', from);
              navigate(from, { replace: true });
            }} />
          } 
        />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute isLoggedIn={isLoggedIn}>
              {renderContent(
                <Box sx={{ display: 'flex' }}>
                  <CssBaseline />
                  <Navbar 
                    onMenuClick={() => setIsMobileSidebarOpen(true)}
                    onLogout={handleLogout}
                  />
                  <Sidebar 
                    isOpen={isSidebarOpen} 
                    isMobileOpen={isMobileSidebarOpen}
                    onClose={() => setIsMobileSidebarOpen(false)}
                  />
                  <Box component="main" sx={{ flexGrow: 1, p: 3, width: { sm: `calc(100% - ${isSidebarOpen ? 240 : 0}px)` } }}>
                    <Dashboard />
                  </Box>
                </Box>
              )}
            </PrivateRoute>
          }
        />
        <Route
          path="/*"
          element={
            <PrivateRoute isLoggedIn={isLoggedIn}>
              {renderContent(
                <Box sx={{ display: 'flex' }}>
                  <CssBaseline />
                  <Navbar 
                    onMenuClick={() => setIsMobileSidebarOpen(true)}
                    onLogout={handleLogout}
                  />
                  <Sidebar 
                    isOpen={isSidebarOpen} 
                    isMobileOpen={isMobileSidebarOpen}
                    onClose={() => setIsMobileSidebarOpen(false)}
                  />
                  <Box component="main" sx={{ flexGrow: 1, p: 3, width: { sm: `calc(100% - ${isSidebarOpen ? 240 : 0}px)` } }}>
                    <Routes>
                      <Route path="/supply-chain" element={<SupplyChain />} />
                      <Route path="/simulation" element={<Simulation />} />
                      <Route path="/analysis" element={<Analysis />} />
                      <Route path="/settings" element={<Settings />} />
                      <Route path="/games" element={<GamesList />} />
                      <Route path="/mixed-games" element={<MixedGamesList />} />
                      <Route path="/create-mixed-game" element={<CreateMixedGame />} />
                      <Route path="/games/:id" element={<GameBoard />} />
                      <Route path="/admin/users" element={<UserManagement />} />
                      <Route path="/" element={<Navigate to="/dashboard" replace />} />
                      <Route path="*" element={<Navigate to="/dashboard" replace />} />
                    </Routes>
                  </Box>
                </Box>
              )}
            </PrivateRoute>
          }
        />
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
