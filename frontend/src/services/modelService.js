import api from './api';

/**
 * Get the status of the GNN model
 * @returns {Promise<Object>} Model status information
 */
export const getModelStatus = async () => {
  try {
    const response = await api.get('/model/status');
    return response.data;
  } catch (error) {
    console.error('Error fetching model status:', error);
    return {
      is_trained: false,
      message: 'Failed to fetch model status',
      error: error.message
    };
  }
};

/**
 * Check if the GNN model is trained
 * @returns {Promise<boolean>} True if model is trained, false otherwise
 */
export const isModelTrained = async () => {
  try {
    const status = await getModelStatus();
    return status.is_trained === true;
  } catch (error) {
    console.error('Error checking if model is trained:', error);
    return false;
  }
};

// Using named exports only for better tree-shaking and consistency
export { getModelStatus, isModelTrained };
