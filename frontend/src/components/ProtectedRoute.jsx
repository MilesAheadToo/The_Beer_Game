import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Box, CircularProgress } from '@mui/material';
import { buildLoginRedirectPath } from '../utils/authUtils';

// Unified ProtectedRoute with optional role checks and children support
function ProtectedRoute({ children, allowedRoles = [] }) {
  const { isAuthenticated, loading, hasAnyRole } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to={buildLoginRedirectPath(location)} replace />;
  }

  if (allowedRoles.length > 0 && !hasAnyRole(allowedRoles)) {
    return <Navigate to="/unauthorized" state={{ from: location }} replace />;
  }

  // If children are provided, render them; otherwise render nested routes
  return children ? children : <Outlet />;
}

export default ProtectedRoute;
