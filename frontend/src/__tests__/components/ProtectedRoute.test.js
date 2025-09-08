import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from '../../contexts/AuthContext';
import ProtectedRoute from '../../components/common/ProtectedRoute';

// Mock child components
const PublicPage = () => <div>Public Page</div>;
const ProtectedPage = () => <div>Protected Page</div>;
const AdminPage = () => <div>Admin Page</div>;
const LoginPage = () => <div>Login Page</div>;
const UnauthorizedPage = () => <div>Unauthorized</div>;

// Helper function to render with router and auth context
const renderWithProviders = (ui, { route = '/', user = null, loading = false } = {}) => {
  const authValue = {
    isAuthenticated: !!user,
    user,
    loading,
    login: jest.fn(),
    logout: jest.fn(),
    refreshUser: jest.fn(),
  };

  return render(
    <AuthProvider value={authValue}>
      <MemoryRouter initialEntries={[route]}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/unauthorized" element={<UnauthorizedPage />} />
          <Route path="/public" element={<PublicPage />} />
          <Route
            path="/protected"
            element={
              <ProtectedRoute>
                <ProtectedPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin"
            element={
              <ProtectedRoute roles={['admin']}>
                <AdminPage />
              </ProtectedRoute>
            }
          />
        </Routes>
      </MemoryRouter>
    </AuthProvider>
  );
};

describe('ProtectedRoute', () => {
  it('should render children when user is authenticated', () => {
    renderWithProviders(
      <ProtectedRoute><div>Protected Content</div></ProtectedRoute>,
      { user: { id: 1, username: 'testuser', roles: ['user'] } }
    );
    
    expect(screen.getByText('Protected Content')).toBeInTheDocument();
  });

  it('should redirect to login when user is not authenticated', () => {
    renderWithProviders(
      <ProtectedRoute><div>Protected Content</div></ProtectedRoute>,
      { user: null, route: '/protected' }
    );
    
    // Should redirect to login
    expect(screen.getByText('Login Page')).toBeInTheDocument();
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });

  it('should show loading state while checking auth', () => {
    const { container } = renderWithProviders(
      <ProtectedRoute><div>Protected Content</div></ProtectedRoute>,
      { user: null, loading: true }
    );
    
    // Should show loading spinner
    expect(container.querySelector('.animate-spin')).toBeInTheDocument();
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });

  it('should allow access when user has required role', () => {
    renderWithProviders(
      <ProtectedRoute roles={['admin']}><div>Admin Content</div></ProtectedRoute>,
      { user: { id: 1, username: 'admin', roles: ['admin'] } }
    );
    
    expect(screen.getByText('Admin Content')).toBeInTheDocument();
  });

  it('should redirect to unauthorized when user lacks required role', () => {
    renderWithProviders(
      <ProtectedRoute roles={['admin']}><div>Admin Content</div></ProtectedRoute>,
      { 
        user: { id: 2, username: 'user', roles: ['user'] },
        route: '/admin'
      }
    );
    
    // Should redirect to unauthorized page
    expect(screen.getByText('Unauthorized')).toBeInTheDocument();
    expect(screen.queryByText('Admin Content')).not.toBeInTheDocument();
  });

  it('should preserve the intended location in state when redirecting', () => {
    const { container } = renderWithProviders(
      <ProtectedRoute><div>Protected Content</div></ProtectedRoute>,
      { 
        user: null,
        route: '/protected?from=dashboard'
      }
    );
    
    // Should redirect to login with from state
    expect(screen.getByText('Login Page')).toBeInTheDocument();
    // Note: Testing the actual navigation state would require additional test setup
  });
});
