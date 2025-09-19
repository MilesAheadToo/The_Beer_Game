import React from 'react';
import { Navigate } from 'react-router-dom';
import { Alert, Box, CircularProgress } from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { isGroupAdmin as isGroupAdminUser, isSystemAdmin as isSystemAdminUser } from '../../utils/authUtils';
import SupplyChainConfigList from '../../components/supply-chain-config/SupplyChainConfigList';
import { TrainingPanel } from './Training';

const GroupSupplyChainConfigList = () => {
  const { user, loading } = useAuth();
  const isSystemAdmin = isSystemAdminUser(user);
  const canAccess = user?.is_superuser || isGroupAdminUser(user);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

  if (!canAccess) {
    return <Navigate to="/unauthorized" replace />;
  }

  const rawGroupId = user?.group_id;
  const parsedGroupId = typeof rawGroupId === 'number' ? rawGroupId : Number(rawGroupId);
  const restrictToGroupId = Number.isFinite(parsedGroupId) ? parsedGroupId : null;

  if (!isSystemAdmin && isGroupAdminUser(user) && !restrictToGroupId) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <Alert severity="warning">
          You must be assigned to a group before you can manage supply chain configurations.
        </Alert>
      </Box>
    );
  }

  return (
    <Box display="flex" flexDirection="column" gap={4}>
      <SupplyChainConfigList
        title="My Group's Supply Chain Configurations"
        basePath="/admin/group/supply-chain-configs"
        restrictToGroupId={isSystemAdmin ? null : restrictToGroupId}
        enableTraining
      />
      <TrainingPanel />
    </Box>
  );
};

export default GroupSupplyChainConfigList;
