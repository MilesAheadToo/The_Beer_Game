import { useEffect } from 'react';
import { useLocation, Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { toast } from 'react-toastify';

/**
 * ProtectedRoute component that redirects to login if not authenticated
 * and checks for required roles if specified.
 * 
 * @param {Object} props - Component props
 * @param {string[]} [props.allowedRoles] - Array of allowed role names (optional)
 * @param {React.ReactNode} [props.unauthorized] - Custom unauthorized component (optional)
 * @param {boolean} [props.requireVerifiedEmail=false] - Whether to require email verification
 * @returns {JSX.Element} The protected route content or redirect
 */
const ProtectedRoute = ({
  allowedRoles = [],
  unauthorized: UnauthorizedComponent,
  requireVerifiedEmail = false,
  ...rest
}) => {
  const { user, isAuthenticated, loading, hasAnyRole } = useAuth();
  const location = useLocation();

  // Check if user has required roles
  const hasRequiredRole = !allowedRoles.length || hasAnyRole(allowedRoles);
  
  // Check if email verification is required and user's email is verified
  const isEmailVerified = !requireVerifiedEmail || (user && user.is_email_verified);

  // Show loading state while checking auth
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    // Store the attempted URL for redirecting after login
    const from = `${location.pathname}${location.search}`;
    return <Navigate to={`/login?redirect=${encodeURIComponent(from)}`} replace />;
  }

  // Check for required roles
  if (!hasRequiredRole) {
    if (UnauthorizedComponent) {
      return <UnauthorizedComponent />;
    }
    
    toast.error('You do not have permission to access this page');
    return <Navigate to="/unauthorized" replace />;
  }

  // Check for email verification
  if (!isEmailVerified) {
    toast.warning('Please verify your email address to continue');
    return <Navigate to="/verify-email" state={{ from: location }} replace />;
  }

  // If all checks pass, render the child routes
  return <Outlet {...rest} />;
};

export default ProtectedRoute;
