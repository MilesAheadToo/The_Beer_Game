import axios from 'axios';
import { API_BASE_URL } from './api';

// Create axios instance for auth endpoints
const authApi = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
  },
});

/**
 * Login with username/email and password
 * @param {Object} credentials - Login credentials
 * @param {string} credentials.username - Username or email
 * @param {string} credentials.password - Password
 * @returns {Promise<Object>} User data
 */
export async function login({ username, password }) {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);
  params.append('grant_type', 'password');

  const response = await authApi.post('/auth/login', params);
  
  // The access token is now in an HTTP-only cookie
  // We return the user data from the response
  return response.data.user;
}

/**
 * Logout the current user
 * @returns {Promise<void>}
 */
export async function logout() {
  await authApi.post('/auth/logout');
}

/**
 * Get current user data
 * @returns {Promise<Object>} User data
 */
export async function getCurrentUser() {
  const response = await authApi.get('/auth/me');
  return response.data;
}

/**
 * Change user password
 * @param {string} currentPassword - Current password
 * @param {string} newPassword - New password
 * @returns {Promise<void>}
 */
export async function changePassword(currentPassword, newPassword) {
  await authApi.post('/auth/change-password', {
    current_password: currentPassword,
    new_password: newPassword,
  });
}

export default {
  login,
  logout,
  getCurrentUser,
  changePassword,
};
