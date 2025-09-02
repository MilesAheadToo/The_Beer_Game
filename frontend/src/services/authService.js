const API_URL = 'http://localhost:8000/api/v1';

export const login = async (email, password) => {
  const response = await fetch(`${API_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'accept': 'application/json'
    },
    body: new URLSearchParams({
      'username': email,
      'password': password,
      'grant_type': 'password'
    })
  });

  if (!response.ok) {
    throw new Error('Login failed');
  }

  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('token_type', data.token_type);
  return data;
};

export const logout = () => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('token_type');
};

export const getAuthHeader = () => {
  const token = localStorage.getItem('access_token');
  const tokenType = localStorage.getItem('token_type') || 'Bearer';
  return token ? `${tokenType} ${token}` : '';
};

export const isAuthenticated = () => {
  return !!localStorage.getItem('access_token');
};
