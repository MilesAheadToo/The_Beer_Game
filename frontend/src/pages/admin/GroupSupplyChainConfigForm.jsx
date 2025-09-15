import React from 'react';
import { Navigate } from 'react-router-dom';
import { Alert, Box, CircularProgress } from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { isGroupAdmin as isGroupAdminUser } from '../../utils/authUtils';
import SupplyChainConfigForm from '../../components/supply-chain-config/SupplyChainConfigForm';

const GroupSupplyChainConfigForm = () => {
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

  const groupId = user?.group_id ?? null;

  if (isGroupAdminUser(user) && !groupId) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <Alert severity="warning">
          You must be assigned to a group before you can create or edit supply chain configurations.
        </Alert>
      </Box>
    );
  }

  return (
    <SupplyChainConfigForm
      basePath="/admin/group/supply-chain-configs"
      allowGroupSelection={false}
      defaultGroupId={groupId}
    />
  );
};

export default GroupSupplyChainConfigForm;
