import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { login } from '../services/authService';
import daybreakLogo from '../assets/daybreak_logo.png';
import beerGameDiagram from '../assets/beer-game-diagram.svg'; // Make sure to add this asset
import { FaEye, FaEyeSlash, FaSignInAlt } from 'react-icons/fa';
import { motion } from 'framer-motion';
import { toast } from 'react-toastify';

function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  // Check if user is already logged in
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      navigate('/dashboard');
    }
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    
    try {
      await login(email, password);
      onLogin();
      navigate(location.state?.from?.pathname || '/dashboard');
    } catch (err) {
      setError('Incorrect email or password. Please try again.');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-daybreak-blue-900 to-daybreak-navy-900 p-4">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 bg-white rounded-2xl shadow-2xl overflow-hidden"
      >
        {/* Left Panel - Beer Game Diagram */}
        <div className="hidden lg:flex flex-col items-center justify-center p-8 bg-gradient-to-br from-daybreak-blue-600 to-daybreak-blue-800 text-white">
          <div className="mb-8 text-center">
            <img 
              src={daybreakLogo} 
              alt="Daybreak Logo" 
              className="h-16 w-auto mx-auto mb-6" 
            />
            <h1 className="text-3xl font-bold mb-2">The Beer Game</h1>
            <p className="text-daybreak-blue-100">Experience the classic supply chain management simulation</p>
          </div>
          
          <div className="w-full max-w-md">
            <img 
              src={beerGameDiagram} 
              alt="Beer Game Supply Chain" 
              className="w-full h-auto opacity-90"
            />
          </div>
          
          <div className="mt-8 text-center text-daybreak-blue-100">
            <p>Learn supply chain dynamics through this interactive simulation</p>
          </div>
        </div>
        
        {/* Right Panel - Login Form */}
        <div className="p-8 sm:p-12 flex flex-col justify-center">
          {/* Mobile Logo */}
          <div className="lg:hidden mb-8 text-center">
            <img 
              src={daybreakLogo} 
              alt="Daybreak Logo" 
              className="h-12 w-auto mx-auto mb-4" 
            />
            <h1 className="text-2xl font-bold text-gray-800">The Beer Game</h1>
          </div>
          
          <div className="max-w-md mx-auto w-full">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome Back</h2>
            <p className="text-gray-600 mb-8">Sign in to continue to Daybreak's Beer Game</p>
            
            {error && (
              <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-lg text-sm">
                {error}
              </div>
            )}
            
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                  Email Address
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-daybreak-blue-500 focus:border-daybreak-blue-500 transition-colors"
                  placeholder="Enter your email"
                  required
                />
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                    Password
                  </label>
                  <a href="/forgot-password" className="text-sm text-daybreak-blue-600 hover:text-daybreak-blue-800 transition-colors">
                    Forgot password?
                  </a>
                </div>
                <div className="relative">
                  <input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-daybreak-blue-500 focus:border-daybreak-blue-500 pr-12 transition-colors"
                    placeholder="Enter your password"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600 transition-colors"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <FaEyeSlash /> : <FaEye />}
                  </button>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    className="h-4 w-4 text-daybreak-blue-600 focus:ring-daybreak-blue-500 border-gray-300 rounded"
                  />
                  <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
                    Remember me
                  </label>
                </div>
              </div>
              
              <div>
                <motion.button
                  type="submit"
                  disabled={isLoading}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-daybreak-blue-600 hover:bg-daybreak-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-daybreak-blue-500 transition-colors ${
                    isLoading ? 'opacity-70 cursor-not-allowed' : ''
                  }`}
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Signing in...
                    </>
                  ) : (
                    <>
                      <FaSignInAlt className="mr-2" />
                      Sign In
                    </>
                  )}
                </motion.button>
              </div>
            </form>
            
            <div className="mt-8 text-center text-sm text-gray-500">
              <p>Don't have an account?{' '}
                <a href="/signup" className="font-medium text-daybreak-blue-600 hover:text-daybreak-blue-500 transition-colors">
                  Contact administrator
                </a>
              </p>
            </div>
            
            <div className="mt-8 pt-6 border-t border-gray-200">
              <p className="text-xs text-gray-500 text-center">
                &copy; {new Date().getFullYear()} Daybreak Game Studios. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      </motion.div>
      
      {/* Background elements */}
      <div className="fixed inset-0 overflow-hidden -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-daybreak-blue-900/30 to-daybreak-navy-900/30"></div>
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
      </div>
      
      <div className="mt-8 text-center text-sm text-gray-400">
        <p>© {new Date().getFullYear()} Daybreak AI. All rights reserved.</p>
      </div>
    </div>
  );
}

export default Login;
