import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AuthProvider } from "./contexts/AuthContext";
import { HelpProvider } from "./contexts/HelpContext";
import { HelmetProvider } from 'react-helmet-async';
import App from './App';
import './index.css';

const theme = createTheme({
  palette: {
    primary: {
      main: "#1976d2",
    },
    secondary: {
      main: "#dc004e",
    },
    background: {
      default: "#f5f5f5",
      paper: "#ffffff",
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 500,
    },
    h2: {
      fontWeight: 500,
    },
    h3: {
      fontWeight: 500,
    },
  },
});

// Create root once
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element not found');
}
const root = createRoot(rootElement);

// Test API connection
async function testApiConnection() {
  console.log('Attempting to connect to API at:', '/api/health');
  try {
    const response = await fetch('/api/health', {
      method: 'GET',
      credentials: 'include',
      headers: {
        'Accept': 'application/json',
        'Cache-Control': 'no-cache',
      },
    });
    
    console.log('API Response Status:', response.status, response.statusText);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error Response:', {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries()),
        body: errorText
      });
      throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
    }
    
    const data = await response.json().catch(e => {
      console.error('Error parsing JSON response:', e);
      return { status: 'success', message: 'Health check passed (no JSON response)' };
    });
    
    console.log('API Health Response:', data);
    return data;
  } catch (error) {
    console.error('API connection test failed:', {
      name: error.name,
      message: error.message,
      stack: error.stack,
      cause: error.cause
    });
    return { 
      status: 'error', 
      message: error.message || 'Failed to connect to API',
      details: error.toString(),
      timestamp: new Date().toISOString()
    };
  }
}

// Debug component to show what's happening
function DebugInfo() {
  const [step, setStep] = React.useState('Initializing...');
  const [error, setError] = React.useState(null);
  const [apiStatus, setApiStatus] = React.useState(null);

  const renderDebugInfo = (status) => {
    root.render(
      <div style={{
        padding: '20px',
        fontFamily: 'Arial, sans-serif',
        maxWidth: '800px',
        margin: '0 auto',
        lineHeight: '1.6'
      }}>
        <h1 style={{ color: '#1976d2' }}>The Beer Game - Debug Info</h1>
        <div style={{
          backgroundColor: '#f8f9fa',
          padding: '20px',
          borderRadius: '5px',
          margin: '20px 0',
          borderLeft: '4px solid #1976d2'
        }}>
          <h3>Debug Information:</h3>
          <ul>
            <li>✓ {step}</li>
            {status && (
              <li>✓ API Status: {JSON.stringify(status)}</li>
            )}
          </ul>
        </div>
        <div style={{ marginTop: '20px' }}>
          <h3>Next Steps:</h3>
          <ol>
            <li>Loading application components...</li>
            <li>Checking authentication status...</li>
          </ol>
        </div>
      </div>
    );
  };

  const renderError = (errorMsg) => {
    root.render(
      <div style={{
        padding: '20px',
        backgroundColor: '#ffebee',
        border: '1px solid #f44336',
        borderRadius: '4px',
        margin: '20px',
        fontFamily: 'monospace',
        whiteSpace: 'pre-wrap'
      }}>
        <h2>Error</h2>
        <div>{errorMsg}</div>
        <div style={{ marginTop: '20px' }}>
          <h4>Current Step:</h4>
          <div>{step}</div>
        </div>
      </div>
    );
  };

  const renderApp = () => {
    root.render(
      <React.StrictMode>
        <HelmetProvider>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <BrowserRouter>
              <AuthProvider>
                <HelpProvider>
                  <App />
                </HelpProvider>
              </AuthProvider>
            </BrowserRouter>
          </ThemeProvider>
        </HelmetProvider>
      </React.StrictMode>
    );
  };

  React.useEffect(() => {
    const init = async () => {
      try {
        setStep('Root element found');
        renderDebugInfo();

        // Check if React is loaded
        if (typeof React === 'undefined' || !React.createElement) {
          throw new Error('React is not loaded');
        }
        setStep('React is loaded');
        renderDebugInfo();

        // Test API connection
        setStep('Testing API connection...');
        renderDebugInfo();
        
        const status = await testApiConnection();
        setApiStatus(status);
        
        if (status.status !== 'ok') {
          throw new Error(`API connection failed: ${status.message || 'Unknown error'}`);
        }
        
        setStep('API connection successful');
        renderDebugInfo(status);

        // After a short delay, render the full app
        setTimeout(() => {
          try {
            setStep('Rendering application...');
            renderDebugInfo(status);
            renderApp();
          } catch (err) {
            console.error('Error rendering app:', err);
            renderError(`Failed to render app: ${err.message}`);
          }
        }, 2000);
      } catch (err) {
        console.error('Initialization error:', err);
        renderError(`Initialization failed: ${err.message}`);
      }
    };

    init();
  }, []);

  return null; // This component doesn't render anything itself
}

// Initial render with debug info
root.render(<DebugInfo />);
