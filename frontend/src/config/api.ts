// /frontend/src/config/api.ts
// Single source of truth for the API origin + base path.

// Default values for development
const DEFAULT_ORIGIN = 'http://localhost:8000';
const DEFAULT_BASE_PATH = '/api/v1';

// Get environment variables with Create React App prefix
const getEnvVar = (key: string, fallback: string = ''): string => {
  // In Create React App, environment variables are available under process.env.REACT_APP_*
  const value = process.env[`REACT_APP_${key}`];
  return value !== undefined ? value : fallback;
};

// Get the origin and base path from environment variables or use defaults
const rawOrigin = getEnvVar('API_ORIGIN', DEFAULT_ORIGIN);
const rawBasePath = getEnvVar('API_BASE_PATH', DEFAULT_BASE_PATH);

// Ensure the origin doesn't end with a slash and base path starts with one
export const API_ORIGIN = rawOrigin.replace(/\/+$/, '');
export const API_BASE_PATH = rawBasePath.startsWith('/') ? rawBasePath : `/${rawBasePath}`;

// For Create React App, we can use relative URLs when the API is on the same origin
export const API_BASE_URL = API_BASE_PATH;
