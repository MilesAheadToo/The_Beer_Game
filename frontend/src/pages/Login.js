import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { login, isAuthenticated } from '../services/authService';
import { FaEye, FaEyeSlash, FaSignInAlt, FaSpinner } from 'react-icons/fa';
import { motion } from 'framer-motion';
import { toast } from 'react-toastify';
import './Login.css';

const daybreakLogo = '/daybreak_logo.png';

function Login({ onLogin }) {
  // Prevent console clearing
  if (window.console) {
    if (window.console.clear) {
      console.log('Console clearing is disabled for debugging');
      window.console.clear = function() {
        console.log('Console clear was prevented');
      };
    }
  }
  const [email, setEmail] = useState('admin@daybreak.ai');
  const [password, setPassword] = useState('Daybreak@2025');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  // Check if user is already logged in
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      console.log('Found access token, redirecting to dashboard');
      const from = location.state?.from?.pathname || '/dashboard';
      navigate(from, { replace: true });
    }
  }, [navigate, location.state]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
  
    try {
      // clear (optional)
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
  
      const loginResponse = await login(email, password);
  
      // confirm token exists
      const token = localStorage.getItem('access_token');
      if (!token) throw new Error('No token found in localStorage after login');
    
      // fire-and-forget the UI callback, don't block navigation
      onLogin?.();
  
      // go where they came from, or dashboard
      const from = location.state?.from?.pathname || '/dashboard';
      navigate(from, { replace: true });
    } catch (err) {
      setError(err.message || 'Incorrect email or password. Please try again.');
      setIsLoading(false);
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
    }
  };
  
  return (
    <div className="login-container">
      <header className="login-header">
        <img src={daybreakLogo} alt="Daybreak Logo" className="login-logo" />
        <h1 className="login-title">Daybreak Beer Game with Agentic AI</h1>
      </header>

      <main className="login-content">
        <div className="login-bg-pattern"></div>
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="login-card"
        >
          <div className="login-welcome">
            <h1>Welcome Back</h1>
            <p>Sign in to continue to your account</p>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} autoComplete="on">
            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <div className="input-group">
                <input
                  type="email"
                  id="email"
                  name="email"
                  className="form-control"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email"
                  autoComplete="username"
                  autoCapitalize="off"
                  autoCorrect="off"
                  spellCheck="false"
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <div className="input-group password-input">
                <input
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  name="password"
                  className="form-control"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  autoCapitalize="off"
                  autoCorrect="off"
                  spellCheck="false"
                  required
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <FaEyeSlash /> : <FaEye />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              className="btn-login"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <FaSpinner className="spinner" />
                  Signing in...
                </>
              ) : (
                <>
                  <FaSignInAlt style={{ marginRight: '8px' }} />
                  Sign In
                </>
              )}
            </button>
          </form>

          <div className="login-footer">
            <p>Don't have an account? Contact your administrator</p>
          </div>
        </motion.div>
      </main>
    </div>
  );
}

export default Login;
