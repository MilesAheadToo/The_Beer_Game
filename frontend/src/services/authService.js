const API_URL = 'http://localhost:8000/api/v1';

// Helper function to log to both console and page
const debugLog = (...args) => {
  console.log(...args);
  // Log to a visible element on the page
  const debugEl = document.getElementById('debug-log') || (() => {
    const el = document.createElement('div');
    el.id = 'debug-log';
    el.style.position = 'fixed';
    el.style.top = '10px';
    el.style.right = '10px';
    el.style.backgroundColor = 'rgba(0,0,0,0.8)';
    el.style.color = '#fff';
    el.style.padding = '10px';
    el.style.maxWidth = '400px';
    el.style.maxHeight = '300px';
    el.style.overflow = 'auto';
    el.style.zIndex = '9999';
    el.style.fontFamily = 'monospace';
    el.style.fontSize = '12px';
    document.body.appendChild(el);
    return el;
  })();
  
  const logEntry = document.createElement('div');
  logEntry.textContent = args.map(arg => 
    typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
  ).join(' ');
  
  debugEl.appendChild(logEntry);
  debugEl.scrollTop = debugEl.scrollHeight;
};

export const login = async (email, password) => {
  debugLog('1. Starting login process...', { email });
  
  try {
    const loginUrl = `${API_URL}/auth/login`;
    debugLog('2. Sending login request to:', loginUrl);
    
    debugLog('3. Sending login request with body:', { username: email, grant_type: 'password' });
    
    const response = await fetch(loginUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
      },
      body: new URLSearchParams({
        'username': email,
        'password': password,
        'grant_type': 'password',
      }),
      #credentials: 'include'
    });
    
    debugLog('4. Received response status:', response.status);
    debugLog('5. Response headers:', Object.fromEntries([...response.headers.entries()]));
    
    const responseText = await response.text();
    debugLog('6. Raw response text:', responseText);
    
    let data;
    try {
      data = responseText ? JSON.parse(responseText) : {};
      debugLog('7. Parsed response data:', data);
      
      if (!response.ok) {
        debugLog('8. Login failed with status:', response.status);
        throw new Error(data.detail || 'Login failed');
      }
      
      if (!data.access_token) {
        debugLog('9. No access token in response');
        throw new Error('No access token received from server');
      }
      
      // Store tokens
      debugLog('10. Storing tokens in localStorage');
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem(
        'token_type',
        (data.token_type || 'Bearer').replace(/^bearer$/i, 'Bearer')
      );
      
      // Verify storage
      const storedToken = localStorage.getItem('access_token');
      debugLog('11. Token storage verification:', {
        stored: !!storedToken,
        length: storedToken?.length || 0,
        start: storedToken?.substring(0, 10) + '...' || 'N/A'
      });
      
      if (data.user) {
        debugLog('12. Storing user data');
        localStorage.setItem('user', JSON.stringify(data.user));
      }
      
      return data;
      
    } catch (e) {
      debugLog('ERROR:', e.message);
      throw e;
    }
    
    return data;
  } catch (error) {
    console.error('Error during login:', error);
    throw error;
  }
};

export const logout = () => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('token_type');
};

export const getAuthHeader = () => {
  const token = localStorage.getItem('access_token');
  const tokenType = (localStorage.getItem('token_type') || 'Bearer').replace(/^bearer$/i, 'Bearer');
  return token ? `${tokenType} ${token}` : '';
};

export const isAuthenticated = async () => {
  console.log('A1. Starting isAuthenticated check...');
  const token = localStorage.getItem('access_token');
  const tokenType = (localStorage.getItem('token_type') || 'Bearer').replace(/^bearer$/i, 'Bearer');
  
  console.log('A2. Token in localStorage:', {
    hasToken: !!token,
    tokenType,
    tokenLength: token?.length || 0,
    tokenStart: token ? token.substring(0, 10) + '...' : 'N/A'
  });
  
  if (!token) {
    console.log('A3. No token found in localStorage');
    return false;
  }

  try {
    console.log('A4. Sending token validation request to backend...');
    const response = await fetch(`${API_URL}/auth/me`, {
      method: 'GET',
      headers: {
        'Authorization': `${tokenType} ${token}`,
        'Accept': 'application/json',
      },
      #credentials: 'include'
    });

    console.log('A5. Received response from /auth/me:', {
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries([...response.headers.entries()])
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('A6. Token validation failed with status:', response.status, 'Response:', errorText);
      // Clear invalid token
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
      return false;
    }

    console.log('A6. Token is valid');
    return true;
  } catch (error) {
    console.error('Error validating token:', error);
    return false;
  }
};
