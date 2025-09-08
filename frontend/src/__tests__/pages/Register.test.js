import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Register from '../../pages/Register';
import { AuthProvider } from '../../contexts/AuthContext';

// Mock the API
jest.mock('../../services/api', () => ({
  mixedGameApi: {
    register: jest.fn(),
  },
}));

// Mock the useNavigate hook
const mockNavigate = jest.fn();

jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
  useSearchParams: () => [new URLSearchParams(), jest.fn()],
}));

// Mock react-toastify
global.toast = {
  success: jest.fn(),
  error: jest.fn(),
};

describe('Register', () => {
  const renderRegister = () => {
    return render(
      <MemoryRouter>
        <AuthProvider>
          <Register />
        </AuthProvider>
      </MemoryRouter>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock window.scrollTo
    window.scrollTo = jest.fn();
  });

  it('renders the registration form', () => {
    renderRegister();
    
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/first name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/last name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^password/i)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /terms/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /log in/i })).toBeInTheDocument();
  });

  it('validates form fields', async () => {
    renderRegister();
    
    // Submit empty form
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));
    
    // Check for validation errors
    expect(await screen.findByText(/username is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/email is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/first name is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/last name is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/password is required/i)).toBeInTheDocument();
    expect(await screen.findByText(/you must accept the terms/i)).toBeInTheDocument();
    
    // Test invalid username
    const usernameInput = screen.getByPlaceholderText('Username');
    fireEvent.change(usernameInput, { target: { value: 'ab' } });
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));
    expect(await screen.findByText(/username must be at least 3 characters/i)).toBeInTheDocument();
    
    // Test invalid email
    const emailInput = screen.getByPlaceholderText('Email address');
    fireEvent.change(emailInput, { target: { value: 'invalid-email' } });
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));
    expect(await screen.findByText(/enter a valid email address/i)).toBeInTheDocument();
    
    // Test password mismatch
    const passwordInput = screen.getByPlaceholderText('Password');
    const confirmPasswordInput = screen.getByPlaceholderText('Confirm password');
    fireEvent.change(passwordInput, { target: { value: 'Password123!' } });
    fireEvent.change(confirmPasswordInput, { target: { value: 'Different123!' } });
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));
    expect(await screen.findByText(/passwords do not match/i)).toBeInTheDocument();
  });

  it('handles successful registration', async () => {
    require('../../services/api').mixedGameApi.register.mockResolvedValue({ success: true });
    
    renderRegister();
    
    // Fill in the form
    fireEvent.change(screen.getByPlaceholderText('Username'), {
      target: { value: 'testuser' },
    });
    fireEvent.change(screen.getByPlaceholderText('Email address'), {
      target: { value: 'test@example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText('First name'), {
      target: { value: 'Test' },
    });
    fireEvent.change(screen.getByPlaceholderText('Last name'), {
      target: { value: 'User' },
    });
    fireEvent.change(screen.getByPlaceholderText('Password'), {
      target: { value: 'Password123!' },
    });
    fireEvent.change(screen.getByPlaceholderText('Confirm password'), {
      target: { value: 'Password123!' },
    });
    fireEvent.click(screen.getByRole('checkbox', { name: /terms/i }));
    
    // Submit the form
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));
    
    // Check that the register function was called with the right parameters
    await waitFor(() => {
      expect(require('../../services/api').mixedGameApi.register).toHaveBeenCalledWith({
        username: 'testuser',
        email: 'test@example.com',
        firstName: 'Test',
        lastName: 'User',
        password: 'Password123!',
      });
    });
    
    // Check for success message and redirection
    await waitFor(() => {
      expect(global.toast.success).toHaveBeenCalledWith(
        'Registration successful! Please check your email to verify your account.',
        expect.any(Object)
      );
    });
  });

  it('handles registration failure', async () => {
    const errorMessage = 'Username already exists';
    require('../../services/api').mixedGameApi.register.mockRejectedValue({
      response: { data: { detail: errorMessage } },
    });
    
    renderRegister();
    
    // Fill in the form with minimal valid data
    fireEvent.change(screen.getByPlaceholderText('Username'), { target: { value: 'existinguser' } });
    fireEvent.change(screen.getByPlaceholderText('Email address'), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('First name'), { target: { value: 'Test' } });
    fireEvent.change(screen.getByPlaceholderText('Last name'), { target: { value: 'User' } });
    fireEvent.change(screen.getByPlaceholderText('Password'), { target: { value: 'Password123!' } });
    fireEvent.change(screen.getByPlaceholderText('Confirm password'), { target: { value: 'Password123!' } });
    fireEvent.click(screen.getByRole('checkbox', { name: /terms/i }));
    
    // Submit the form
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));
    
    // Check that the error message is displayed
    await waitFor(() => {
      expect(global.toast.error).toHaveBeenCalledWith(errorMessage, expect.any(Object));
    });
  });

  it('toggles password visibility', () => {
    renderRegister();
    
    const passwordInput = screen.getByPlaceholderText('Password');
    const confirmPasswordInput = screen.getByPlaceholderText('Confirm password');
    const toggleButtons = screen.getAllByRole('button', { name: /show password/i });
    
    // Passwords should be hidden by default
    expect(passwordInput).toHaveAttribute('type', 'password');
    expect(confirmPasswordInput).toHaveAttribute('type', 'password');
    
    // Toggle password visibility for the first password
    fireEvent.click(toggleButtons[0]);
    
    // First password should be visible, second should remain hidden
    expect(passwordInput).toHaveAttribute('type', 'text');
    expect(confirmPasswordInput).toHaveAttribute('type', 'password');
    
    // Toggle password visibility for the confirm password
    fireEvent.click(toggleButtons[1]);
    
    // Both passwords should be visible
    expect(passwordInput).toHaveAttribute('type', 'text');
    expect(confirmPasswordInput).toHaveAttribute('type', 'text');
  });
});
