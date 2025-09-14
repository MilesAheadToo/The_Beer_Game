import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Login from '../../pages/Login.jsx';
import { AuthProvider } from '../../contexts/AuthContext';

// Mock the API
jest.mock('../../services/api', () => {
  const apiMock = {
    login: jest.fn(),
    getGames: jest.fn().mockResolvedValue([]),
    getCurrentUser: jest.fn().mockResolvedValue(null),
    refreshToken: jest.fn(),
    logout: jest.fn(),
  };
  return { __esModule: true, mixedGameApi: apiMock, default: apiMock };
});

// Mock the useNavigate hook
const mockNavigate = jest.fn();

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
  useLocation: () => ({
    state: { from: { pathname: '/dashboard' } },
  }),
}));

// Mock react-toastify
global.toast = {
  error: jest.fn(),
};

describe('Login', () => {
  const renderLogin = () => {
    return render(
      <MemoryRouter>
        <AuthProvider>
          <Login />
        </AuthProvider>
      </MemoryRouter>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the login form', () => {
    renderLogin();
    
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /forgot your password/i })).toBeInTheDocument();
    expect(screen.getByText(/create a new account/i)).toBeInTheDocument();
  });

  it('validates form fields', async () => {
    renderLogin();
    
    // Submit empty form
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    // Check for validation errors
    expect(await screen.findByText(/email is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/password is required/i)).toBeInTheDocument();
    
    // Test invalid email
    const emailInput = screen.getByPlaceholderText('Email address');
    fireEvent.change(emailInput, { target: { value: 'invalid-email' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    expect(await screen.findByText(/enter a valid email address/i)).toBeInTheDocument();
    
    // Test short password
    const passwordInput = screen.getByPlaceholderText('Password');
    fireEvent.change(passwordInput, { target: { value: 'short' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    expect(await screen.findByText(/password must be at least 8 characters/i)).toBeInTheDocument();
  });

  it('handles successful login', async () => {
    const mockUser = { id: 1, username: 'testuser', email: 'test@example.com' };
    require('../../services/api').mixedGameApi.login.mockResolvedValue({ success: true, user: mockUser });
    
    renderLogin();
    
    // Fill in the form
    fireEvent.change(screen.getByPlaceholderText('Email address'), {
      target: { value: 'test@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText('Password'), {
      target: { value: 'password123' },
    });
    
    // Submit the form
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    // Check that the login function was called with the right parameters
    await waitFor(() => {
      expect(require('../../services/api').mixedGameApi.login).toHaveBeenCalledWith({
        username: 'test@example.com',
        password: 'password123',
      });
    });

    // Check that we navigate to the games list after successful login
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/games', { replace: true });
    });
  });

  it('handles login failure', async () => {
    const errorMessage = 'Invalid credentials';
    require('../../services/api').mixedGameApi.login.mockResolvedValue({ success: false, error: errorMessage });
    
    renderLogin();
    
    // Fill in the form
    fireEvent.change(screen.getByPlaceholderText('Email address'), {
      target: { value: 'test@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText('Password'), {
      target: { value: 'wrongpassword' },
    });
    
    // Submit the form
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    
    // Check that the error message is displayed
    await waitFor(() => {
      expect(global.toast.error).toHaveBeenCalledWith(errorMessage, expect.any(Object));
    });
  });
});
