import React, { useEffect, useState } from 'react';
import { Box, Chip, Tooltip } from '@mui/material';
import { api } from '../services/api';
import { API_BASE_URL } from '../config/api.ts';

// Small dev-only banner to display the resolved API base URL.
export default function DebugBanner() {
  const [resolved, setResolved] = useState('');
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const isProd = process.env.NODE_ENV === 'production';
    const enable = !isProd || /[?&]debug=1\b/.test(window.location.search) || localStorage.getItem('DBG_API_BANNER') === '1';
    setVisible(Boolean(enable));
    setResolved(api?.defaults?.baseURL || '');
  }, []);

  if (!visible) return null;

  const text = resolved || '(unset)';
  const hint = `Axios: ${text}\nEnv default: ${API_BASE_URL}`;

  return (
    <Box sx={{ position: 'fixed', top: 8, right: 8, zIndex: 2000 }}>
      <Tooltip title={<pre style={{ margin: 0 }}>{hint}</pre>}>
        <Chip
          size="small"
          color="default"
          variant="outlined"
          label={`API: ${text}`}
          onClick={() => {
            try {
              navigator.clipboard.writeText(text);
            } catch (_) {}
          }}
        />
      </Tooltip>
    </Box>
  );
}

