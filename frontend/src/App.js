import React from 'react';
import { Routes, Route, Navigate, Outlet, useLocation } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import MixedGamesList from './pages/MixedGamesList';
import CreateMixedGame from './pages/CreateMixedGame';
import GameBoard from './pages/GameBoard';
import Login from './pages/Login';
import Users from './pages/Users';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { useAuth } from './contexts/AuthContext';
import HumanDashboard from './pages/HumanDashboard';
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

function RequireAuth({ adminOnly = false }) {
  const { isAuthenticated, loading, user } = useAuth();
  const location = useLocation();
  
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }
  
  if (!isAuthenticated) {
    const back = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?redirect=${back}`} replace />;
  }
  
  // If route requires admin but user is not admin, redirect to human dashboard
  if (adminOnly && !user?.is_admin) {
    return <Navigate to="/dashboard" replace />;
  }
  
  return <Outlet />;
}

const AppContent = () => {
  const location = useLocation();
  const isGamePage = location.pathname.startsWith('/games/');

  return (
    <Box sx={{ display: 'flex' }}>
      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1, p: 3, width: '100%' }}>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          
          {/* Private Routes */}
          {/* Admin-only routes */}
          <Route element={<RequireAuth adminOnly={true} />}>
            <Route path="/admin" element={
              <>
                <Navbar />
                <Dashboard />
              </>
            } />
            
            <Route path="/users" element={
              <>
                <Navbar />
                <Users />
              </>
            } />
            
            <Route path="/games" element={
              <>
                <Navbar />
                <MixedGamesList />
              </>
            } />
            
            <Route path="/games/new" element={
              <>
                <Navbar />
                <CreateMixedGame />
              </>
            } />
          </Route>

          {/* Regular user routes */}
          <Route element={<RequireAuth />}>
            <Route path="/dashboard" element={
              <>
                <Navbar />
                <HumanDashboard />
              </>
            } />
            
            <Route path="/games/:gameId" element={
              isGamePage ? (
                <WebSocketProvider>
                  <Navbar />
                  <GameBoard />
                </WebSocketProvider>
              ) : (
                <>
                  <Navbar />
                  <GameBoard />
                </>
              )
            } />
            
            {/* Default redirect */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            
            {/* 404 - Redirect to dashboard if route not found */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        </Routes>
      </Box>
    </Box>
  );
}

export default AppContent;
