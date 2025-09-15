import React from 'react';
import { Navigate } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { isGroupAdmin as isGroupAdminUser } from '../../utils/authUtils';
import SupplyChainConfigList from '../../components/supply-chain-config/SupplyChainConfigList';

const GroupSupplyChainConfigList = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  const canAccess = user?.is_superuser || isGroupAdminUser(user);
  if (!canAccess) {
    return <Navigate to="/unauthorized" replace />;
  }

  return (
    <SupplyChainConfigList
      title="My Group's Supply Chain Configurations"
      basePath="/admin/group/supply-chain-configs"
    />
  );
};

export default GroupSupplyChainConfigList;
