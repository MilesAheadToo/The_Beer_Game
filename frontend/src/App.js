import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate, Outlet, useLocation } from 'react-router-dom';
import { Box, CircularProgress, CssBaseline, useMediaQuery } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { SnackbarProvider } from 'notistack';
import { useAuth } from './contexts/AuthContext';
import Navbar from './components/Navbar';
import { WebSocketProvider } from './contexts/WebSocketContext';
import daybreakTheme from './theme/daybreakTheme';

// Lazy load pages for better performance
const Login = lazy(() => import('./pages/Login'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const MixedGamesList = lazy(() => import('./pages/MixedGamesList'));
const CreateMixedGame = lazy(() => import('./pages/CreateMixedGame'));
const GameBoard = lazy(() => import('./pages/GameBoard'));
const Users = lazy(() => import('./pages/Users'));
const HumanDashboard = lazy(() => import('./pages/HumanDashboard'));
const NotFound = lazy(() => import('./pages/NotFound'));

// Constants
const drawerWidth = 240;

// Global error handler
window.onerror = function(message, source, lineno, colno, error) {
  console.error('Global error:', { message, source, lineno, colno, error });
  return false; // Don't suppress default error handling
};

// Catch unhandled promise rejections
window.onunhandledrejection = function(event) {
  console.error('Unhandled rejection (promise):', event.reason);
};

// Loading component for lazy loading
const LoadingFallback = () => (
  <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
    <CircularProgress />
  </Box>
);

// Route guard for admin routes
const AdminRoute = ({ children }) => {
  const { user } = useAuth();
  
  if (!user?.is_admin) {
    return <Navigate to="/dashboard" replace />;
  }
  
  return <>{children}</>;
};

// Layout component for authenticated routes
const AuthenticatedLayout = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();
  const isMobile = useMediaQuery(theme => theme.breakpoints.down('md'));

  if (loading) {
    return <LoadingFallback />;
  }

  if (!isAuthenticated) {
    const back = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?redirect=${back}`} replace />;
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <CssBaseline />
      {!isMobile && <Navbar />}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: isMobile ? 2 : 3,
          width: { xs: '100%', md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          transition: theme => theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Suspense fallback={<LoadingFallback />}>
          {children}
        </Suspense>
      </Box>
    </Box>
  );
};

// Main app component
const App = () => {
  return (
    <ThemeProvider theme={daybreakTheme}>
      <SnackbarProvider maxSnack={3}>
        <WebSocketProvider>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              {/* Public Routes */}
              <Route path="/login" element={<Login />} />
              
              {/* Protected Routes */}
              <Route element={<AuthenticatedLayout />}>
                {/* Admin Routes */}
                <Route path="/admin" element={
                  <AdminRoute>
                    <Dashboard />
                  </AdminRoute>
                } />
                
                <Route path="/users" element={
                  <AdminRoute>
                    <Users />
                  </AdminRoute>
                } />
                
                <Route path="/games" element={
                  <AdminRoute>
                    <MixedGamesList />
                  </AdminRoute>
                } />
                
                <Route path="/games/new" element={
                  <AdminRoute>
                    <CreateMixedGame />
                  </AdminRoute>
                } />
                
                {/* Regular User Routes */}
                <Route path="/dashboard" element={<HumanDashboard />} />
                <Route path="/games/:gameId" element={<GameBoard />} />
                
                {/* Catch-all route */}
                <Route path="*" element={<NotFound />} />
              </Route>
            </Routes>
          </Suspense>
        </WebSocketProvider>
      </SnackbarProvider>
    </ThemeProvider>
  );
};

export default App;
