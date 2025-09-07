import { Routes, Route, Navigate, useLocation, Outlet } from 'react-router-dom';
import { Suspense, lazy, useState, useEffect } from 'react';
import { CircularProgress, Box } from '@mui/material';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useAuth } from './contexts/AuthContext';
import { useHelp } from './contexts/HelpContext';
import ProtectedRoute from './components/ProtectedRoute';

// Lazy load components
const Login = lazy(() => import('./pages/Login'));
const Register = lazy(() => import('./pages/Register'));
const GameLobby = lazy(() => import('./pages/GameLobby'));
const GameRoom = lazy(() => import('./pages/GameRoom'));
const Profile = lazy(() => import('./pages/Profile'));
const Settings = lazy(() => import('./pages/Settings'));
const ForgotPassword = lazy(() => import('./pages/ForgotPassword'));
const ResetPassword = lazy(() => import('./pages/ResetPassword'));
const MFASetup = lazy(() => import('./pages/MFASetup'));
const Layout = lazy(() => import('./components/Layout'));
const HelpCenter = lazy(() => import('./components/help/HelpCenter'));

// Simple loading component
function Loading() {
  return (
    <Box 
      display="flex" 
      justifyContent="center" 
      alignItems="center" 
      minHeight="100vh"
    >
      <CircularProgress />
    </Box>
  );
}

function AppRoutes() {
  const location = useLocation();
  const navigate = useNavigate();
  const { isHelpOpen, closeHelp } = useHelp();
  
  // Hide help on route change
  useEffect(() => {
    closeHelp();
  }, [location.pathname, closeHelp]);

  return (
    <div className="min-h-screen bg-gray-50">
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/mfa-setup" element={<MFASetup />} />
        
        {/* Protected routes */}
        <Route element={<ProtectedRoute />}>
          <Route index element={<GameLobby />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/games/:gameId" element={<GameRoom />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
        
        {/* 404 - Keep this at the bottom */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      
      {/* Help Center Modal */}
      {isHelpOpen && <HelpCenter onClose={closeHelp} />}
      
      <ToastContainer 
        position="top-right" 
        autoClose={5000} 
        hideProgressBar={false} 
        newestOnTop 
        closeOnClick 
        rtl={false} 
        pauseOnFocusLoss 
        draggable 
        pauseOnHover 
      />
    </div>
  );
}


// Error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by error boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong. Please refresh the page.</h1>;
    }
    return this.props.children;
  }
}

// Main App component
function App() {
  const [isReady, setIsReady] = useState(false);
  
  useEffect(() => {
    // Small delay to ensure all styles and fonts are loaded
    const timer = setTimeout(() => {
      setIsReady(true);
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);

  if (!isReady) {
    return <Loading />;
  }

  return (
    <div className="app">
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
          <Route path="/reset-password" element={<ResetPassword />} />
          <Route path="/mfa-setup" element={<MFASetup />} />
          
          {/* Protected routes */}
          <Route element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }>
            <Route index element={<GameLobby />} />
            <Route path="profile" element={<Profile />} />
            <Route path="games/:gameId" element={<GameRoom />} />
            <Route path="settings" element={<Settings />} />
          </Route>
          
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
      
      <ToastContainer 
        position="top-right" 
        autoClose={5000} 
        hideProgressBar={false} 
        newestOnTop 
        closeOnClick 
        rtl={false} 
        pauseOnFocusLoss 
        draggable 
        pauseOnHover 
      />
    </div>
  );
}

export default App;
