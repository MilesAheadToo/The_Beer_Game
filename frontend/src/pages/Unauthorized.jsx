import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';

const Unauthorized = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" minHeight="60vh" px={2}>
      <Typography variant="h4" gutterBottom>
        Unauthorized
      </Typography>
      <Typography variant="body1" color="text.secondary" align="center" gutterBottom>
        You don’t have permission to access this page.
      </Typography>
      <Box mt={2}>
        <Button variant="contained" color="primary" onClick={() => navigate('/')} sx={{ mr: 1 }}>
          Go Home
        </Button>
        <Button variant="outlined" onClick={() => navigate('/login', { state: { from: location } })}>
          Login With Different Account
        </Button>
      </Box>
    </Box>
  );
};

export default Unauthorized;

