import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CardHeader, 
  CircularProgress, 
  Divider, 
  FormControl, 
  InputLabel, 
  MenuItem, 
  Select, 
  TextField, 
  Typography 
} from '@mui/material';
import { useSnackbar } from 'notistack';
import { 
  createGameFromConfig as createGameFromConfigService,
  getAllConfigs as getAllConfigsService
} from '../../services/supplyChainConfigService';
import { getModelStatus } from '../../services/modelService';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import * as gameService from '../../services/gameService';

const CreateGameFromConfig = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [configs, setConfigs] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState('');
  const [gameData, setGameData] = useState({
    name: '',
    description: '',
    max_rounds: 52,
    is_public: true
  });
  const [modelStatus, setModelStatus] = useState(null);

  // Load supply chain configurations
  useEffect(() => {
    const fetchConfigs = async () => {
      try {
        const data = await getAllConfigsService();
        setConfigs(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching configurations:', error);
        enqueueSnackbar('Failed to load supply chain configurations', { variant: 'error' });
        setLoading(false);
      }
    };

    fetchConfigs();
  }, [enqueueSnackbar]);

  // Load Daybreak agent model status
  useEffect(() => {
    (async () => {
      try {
        const status = await getModelStatus();
        setModelStatus(status);
      } catch (e) {
        // non-blocking
      }
    })();
  }, []);

  // Update game data when a config is selected
  useEffect(() => {
    if (selectedConfig) {
      const config = configs.find(c => c.id === selectedConfig);
      if (config) {
        setGameData(prev => ({
          ...prev,
          name: `Game - ${config.name}`,
          description: config.description || ''
        }));
      }
    }
  }, [selectedConfig, configs]);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setGameData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedConfig) {
      enqueueSnackbar('Please select a supply chain configuration', { variant: 'warning' });
      return;
    }
    
    if (!gameData.name.trim()) {
      enqueueSnackbar('Please enter a game name', { variant: 'warning' });
      return;
    }
    
    setSubmitting(true);
    
    try {
      // First, create a game configuration from the supply chain config
      const gameConfig = await createGameFromConfigService(selectedConfig, gameData);
      
      if (gameConfig) {
        // Then create the game using the generated configuration
        const newGame = await gameService.createGame({
          ...gameConfig,
          player_assignments: [
            // Default player assignments can be added here or configured by the user
            { role: 'retailer', player_type: 'human' },
            { role: 'wholesaler', player_type: 'ai' },
            { role: 'distributor', player_type: 'ai' },
            { role: 'manufacturer', player_type: 'ai' },
          ]
        });
        enqueueSnackbar('Game created successfully!', { variant: 'success' });
        navigate(`/games/${newGame.id}`);
        return newGame;
      } else {
        throw new Error('Failed to create game configuration');
      }
    } catch (error) {
      console.error('Error creating game:', error);
      enqueueSnackbar(
        error.response?.data?.detail || 'Failed to create game', 
        { variant: 'error' }
      );
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box maxWidth="md" mx="auto" p={2}>
      {modelStatus && !modelStatus.is_trained && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <AlertTitle>Daybreak Agent Not Trained</AlertTitle>
          The Daybreak agent has not yet been trained, so it cannot be used until training completes. You may still select Basic (heuristics) or LLM agents.
        </Alert>
      )}
      <Card>
        <CardHeader 
          title="Create Game from Supply Chain Configuration" 
          subheader="Select a supply chain configuration to create a new game"
        />
        <Divider />
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Box mb={3}>
              <FormControl fullWidth variant="outlined" margin="normal" required>
                <InputLabel id="config-select-label">Supply Chain Configuration</InputLabel>
                <Select
                  labelId="config-select-label"
                  id="config-select"
                  value={selectedConfig}
                  onChange={(e) => setSelectedConfig(e.target.value)}
                  label="Supply Chain Configuration"
                >
                  {configs.map((config) => (
                    <MenuItem key={config.id} value={config.id}>
                      {config.name} 
                      {config.is_active && ' (Active)'}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              
              <TextField
                fullWidth
                margin="normal"
                required
                id="name"
                name="name"
                label="Game Name"
                value={gameData.name}
                onChange={handleInputChange}
                variant="outlined"
              />
              
              <TextField
                fullWidth
                margin="normal"
                id="description"
                name="description"
                label="Description"
                value={gameData.description}
                onChange={handleInputChange}
                variant="outlined"
                multiline
                rows={3}
              />
              
              <Box display="flex" gap={2} mt={2}>
                <TextField
                  type="number"
                  margin="normal"
                  required
                  id="max_rounds"
                  name="max_rounds"
                  label="Max Rounds"
                  value={gameData.max_rounds}
                  onChange={handleInputChange}
                  variant="outlined"
                  inputProps={{ min: 1, max: 1000 }}
                />
                
                <FormControl component="fieldset" margin="normal">
                  <div style={{ display: 'flex', alignItems: 'center', height: '100%' }}>
                    <label>
                      <input
                        type="checkbox"
                        name="is_public"
                        checked={gameData.is_public}
                        onChange={handleInputChange}
                        style={{ marginRight: '8px' }}
                      />
                      <Typography variant="body1" component="span">
                        Public Game
                      </Typography>
                    </label>
                  </div>
                </FormControl>
              </Box>
            </Box>
            
            <Box mt={4} display="flex" justifyContent="flex-end" gap={2}>
              <Button 
                variant="outlined" 
                onClick={() => navigate(-1)}
                disabled={submitting}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={!selectedConfig || submitting}
              >
                {submitting ? 'Creating...' : 'Create Game'}
              </Button>
            </Box>
          </form>
        </CardContent>
      </Card>
      
      {selectedConfig && (
        <Box mt={4}>
          <Card>
            <CardHeader title="Configuration Preview" />
            <Divider />
            <CardContent>
              <Typography variant="body1" color="textSecondary">
                Select a configuration to see a preview of the game settings that will be generated.
              </Typography>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default CreateGameFromConfig;
