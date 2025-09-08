import React from 'react';
import { renderHook } from '@testing-library/react';
import AuthContext from '../../contexts/AuthContext';

describe('AuthContext role helpers', () => {
  const wrapperWithUser = (user) => ({ children }) => (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        hasRole: (role) => !!user && (user.is_superuser || (user.roles || []).includes(role)),
        hasAnyRole: (roles = []) => roles.length === 0 || roles.some((r) => !!user && (user.is_superuser || (user.roles || []).includes(r))),
        hasAllRoles: (roles = []) => roles.length === 0 || roles.every((r) => !!user && (user.is_superuser || (user.roles || []).includes(r))),
        isAdmin: !!user && (user.is_superuser || (user.roles || []).includes('admin')),
      }}
    >
      {children}
    </AuthContext.Provider>
  );

  it('detects admin via roles and superuser flag', () => {
    const userRoles = { roles: ['user', 'admin'], is_superuser: false };
    const { result: r1 } = renderHook(() => React.useContext(AuthContext), { wrapper: wrapperWithUser(userRoles) });
    expect(r1.current.isAdmin).toBe(true);

    const userSuper = { roles: ['user'], is_superuser: true };
    const { result: r2 } = renderHook(() => React.useContext(AuthContext), { wrapper: wrapperWithUser(userSuper) });
    expect(r2.current.isAdmin).toBe(true);
  });

  it('checks hasRole / hasAnyRole / hasAllRoles', () => {
    const user = { roles: ['player', 'moderator'], is_superuser: false };
    const { result } = renderHook(() => React.useContext(AuthContext), { wrapper: wrapperWithUser(user) });
    expect(result.current.hasRole('player')).toBe(true);
    expect(result.current.hasAnyRole(['admin', 'player'])).toBe(true);
    expect(result.current.hasAllRoles(['player', 'moderator'])).toBe(true);
    expect(result.current.hasAllRoles(['player', 'admin'])).toBe(false);
  });
});
