import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { 
  FaEye, 
  FaEyeSlash, 
  FaSignInAlt, 
  FaLock, 
  FaEnvelope
} from 'react-icons/fa';
import { 
  Button, 
  Box, 
  VStack, 
  Input, 
  InputGroup, 
  InputLeftElement,
  InputRightElement, 
  FormControl, 
  FormLabel,
  useColorModeValue,
  Text,
  IconButton
} from '@chakra-ui/react';
import PageLayout from '../components/PageLayout';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';

const daybreakLogo = '/daybreak_logo.png';

function Login() {
  const [email, setEmail] = useState('admin@daybreak.ai');
  const [password, setPassword] = useState('Daybreak@2025');
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const { isAuthenticated, login: setAuthed } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const hasNavigated = useRef(false);

  // if already authed, bounce to redirect or /games ONCE
  useEffect(() => {
    if (hasNavigated.current) return;
    if (isAuthenticated) {
      hasNavigated.current = true;
      const search = new URLSearchParams(location.search);
      const redirectTo = search.get("redirect") || "/games";
      console.log('Redirecting to:', redirectTo);
      navigate(redirectTo, { replace: true });
    }
  }, [isAuthenticated, location.search, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError('');

    try {
      const response = await api.login({ 
        username: email, 
        password,
        grant_type: 'password' 
      });

      setAuthed({
        access_token: response.access_token,
        token_type: response.token_type,
        refresh_token: response.refresh_token
      });

      // Immediate navigate to avoid any race with guards
      const search = new URLSearchParams(location.search);
      navigate(search.get("redirect") || "/games", { replace: true });

    } catch (error) {
      console.error('Login error:', error);
      setError('Invalid email or password');
    } finally {
      setSubmitting(false);
    }
  };
  
  const togglePasswordVisibility = () => setShowPassword((v) => !v);

  return (
    <PageLayout title="Sign In">
      <Box 
        maxW="md" 
        mx="auto" 
        mt={12} 
        p={8} 
        bg={cardBg}
        borderRadius="lg"
        boxShadow="lg"
        borderWidth="1px"
        borderColor={borderColor}
      >
        <VStack spacing={6} align="stretch">
          <Box textAlign="center">
            <img 
              src={daybreakLogo} 
              alt="Daybreak Logo" 
              style={{ 
                height: '80px', 
                margin: '0 auto 16px',
                borderRadius: '8px'
              }} 
            />
            <Text fontSize="2xl" fontWeight="bold" color="blue.600">Daybreak AI</Text>
            <Text color="gray.500">Supply Chain Optimization Platform</Text>
          </Box>
          <Text color="gray.600">Sign in to continue to Daybreak Beer Game</Text>
          <form onSubmit={handleSubmit}>
            <VStack spacing={4}>
              <FormControl isRequired>
                <FormLabel>Email</FormLabel>
                <InputGroup>
                  <InputLeftElement pointerEvents="none">
                    <FaEnvelope color="green.500" />
                  </InputLeftElement>
                  <Input
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    isDisabled={submitting}
                    aria-label="Email"
                    size="md"
                    pl={10}
                  />
                </InputGroup>
              </FormControl>

              <FormControl isRequired>
                <FormLabel>Password</FormLabel>
                <InputGroup>
                  <InputLeftElement pointerEvents="none">
                    <FaLock color="green.500" />
                  </InputLeftElement>
                  <Input
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    isDisabled={submitting}
                    aria-label="Password"
                    size="md"
                    pl={10}
                  />
                  <InputRightElement>
                    <IconButton
                      variant="ghost"
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                      icon={showPassword ? <FaEyeSlash /> : <FaEye />}
                      onClick={() => setShowPassword(!showPassword)}
                      isDisabled={submitting}
                      colorScheme="green"
                      color="green.500"
                      _hover={{ color: 'green.600' }}
                    />
                  </InputRightElement>
                </InputGroup>
              </FormControl>
              {error && (
                <Text color="red.500" fontSize="sm" mb={2}>{error}</Text>
              )}
              <Box display="flex" justifyContent="center" width="100%" mt={4}>
                <Button
                  type="submit"
                  colorScheme="blue"
                  size="md"
                  width="200px"
                  isLoading={submitting}
                  loadingText="Signing In..."
                  leftIcon={<FaSignInAlt />}
                  textTransform="none"
                  fontWeight="500"
                  height="40px"
                  _hover={{
                    transform: 'translateY(-1px)',
                  }}
                  _active={{
                    transform: 'none'
                  }}
                >
                  Sign In
                </Button>
              </Box>
              <Box textAlign="center" mt={4}>
                <Text color="gray.600">
                  Don't have an account?{' '}
                  <Link to="/contact" color="blue.600" _hover={{ textDecoration: 'underline' }}>
                    Contact Admin
                  </Link>
                </Text>
                <Text mt={2}>
                  <Link to="/forgot-password" color="blue.600" _hover={{ color: 'blue.700', textDecoration: 'underline' }}>
                    Forgot Password?
                  </Link>
                </Text>
              </Box>
            </VStack>
          </form>
        </VStack>
      </Box>
    </PageLayout>
  );
}

export default Login;
