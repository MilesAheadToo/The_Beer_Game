import React, { useEffect, useMemo, useState } from 'react';
import { Box, Button, Card, CardBody, CardHeader, FormControl, FormLabel, Grid, Heading, HStack, Input, Select, Spinner, Text, useToast, VStack } from '@chakra-ui/react';
import { AddIcon } from '@chakra-ui/icons';
import { Link } from 'react-router-dom';
import PageLayout from '../components/PageLayout';
import mixedGameApi, { api } from '../services/api';
import gameApi from '../services/gameApi';
import { getAllConfigs as getAllSupplyConfigs } from '../services/supplyChainConfigService';

const roleOptions = [
  { value: 'retailer', label: 'Retailer' },
  { value: 'wholesaler', label: 'Wholesaler' },
  { value: 'distributor', label: 'Distributor' },
  { value: 'factory', label: 'Factory' },
];

const PlayersPage = () => {
  const toast = useToast();
  const [games, setGames] = useState([]);
  const [coreConfigs, setCoreConfigs] = useState([]);
  const [selectedCoreConfigId, setSelectedCoreConfigId] = useState('');
  const [users, setUsers] = useState([]);
  const [selectedGameId, setSelectedGameId] = useState('');
  const [loading, setLoading] = useState(true);
  const [players, setPlayers] = useState([]);

  const [form, setForm] = useState({
    name: '',
    role: 'retailer',
    type: 'agent',
    agent_type: 'LLM_BALANCED',
    user_id: '',
  });

  const selectedGame = useMemo(() => games.find(g => String(g.id) === String(selectedGameId)), [games, selectedGameId]);

  const fetchPlayers = async (gameId) => {
    if (!gameId) return;
    try {
      const data = await mixedGameApi.getPlayers(gameId);
      setPlayers(data);
    } catch (e) {
      console.error(e);
      toast({ title: 'Failed to load players', status: 'error' });
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const [gamesData, usersRes, cfgs] = await Promise.all([
          gameApi.getGames(),
          api.get('/auth/users/'),
          (async () => {
            try {
              const list = await getAllSupplyConfigs();
              return Array.isArray(list) ? list : [];
            } catch {
              return [];
            }
          })()
        ]);
        setGames(gamesData);
        setUsers(usersRes.data || []);
        setCoreConfigs(cfgs);
        if (cfgs?.length) setSelectedCoreConfigId(String(cfgs[0].id));
        if (Array.isArray(gamesData) && gamesData.length) {
          setSelectedGameId(String(gamesData[0].id));
        }
      } catch (e) {
        console.error(e);
        toast({ title: 'Failed to load initial data', status: 'error' });
      } finally {
        setLoading(false);
      }
    })();
  }, [toast]);

  useEffect(() => { fetchPlayers(selectedGameId); }, [selectedGameId]);

  const handleAdd = async () => {
    if (!selectedGameId) return;
    try {
      const payload = {
        name: form.name || `${form.role} (${form.type === 'human' ? 'Human' : 'Agent'})`,
        role: form.role,
        is_ai: form.type !== 'human',
        player_type: form.type === 'human' ? 'human' : 'agent',
        agent_type: form.type !== 'human' ? form.agent_type : undefined,
        user_id: form.type === 'human' ? Number(form.user_id) || null : null,
      };
      if (payload.agent_type === 'DAYBREAK_DTCE_CENTRAL') {
        payload.daybreak_override_pct = 0.05;
      }
      await mixedGameApi.addPlayer(Number(selectedGameId), payload);
      toast({ title: 'Player added', status: 'success' });
      setForm(prev => ({ ...prev, name: '' }));
      fetchPlayers(selectedGameId);
    } catch (e) {
      console.error(e);
      const msg = e?.response?.data?.detail || 'Failed to add player';
      toast({ title: 'Error', description: msg, status: 'error' });
    }
  };
  
  // Quick create a minimal game if none exist
  const quickCreateGame = async () => {
    try {
      const payload = {
        name: `Quick Game ${new Date().toLocaleString()}`,
        max_rounds: 20,
        demand_pattern: { type: 'classic', params: { initial_demand: 4, change_week: 6, final_demand: 8 } },
      };
      const newGame = await gameApi.createGame(payload);
      toast({ title: 'Game created', status: 'success' });
      setGames((prev) => (Array.isArray(prev) ? [...prev, newGame] : [newGame]));
      const id = String(newGame.id);
      setSelectedGameId(id);
      fetchPlayers(id);
    } catch (e) {
      console.error('Quick create failed', e);
      toast({ title: 'Failed to create game', description: e?.response?.data?.detail || e.message, status: 'error' });
    }
  };

  if (loading) {
    return (
      <PageLayout title="Players">
        <Box p={8}><Spinner /></Box>
      </PageLayout>
    );
  }

  return (
    <PageLayout title="Players">
      <HStack justify="space-between" mb={6}>
        <Heading size="lg">Players</Heading>
        <HStack>
          <Button as={Link} to="/games" variant="outline">Back to Games</Button>
        </HStack>
      </HStack>

      <Card mb={6}>
        <CardHeader>
          <HStack spacing={4}>
            <FormControl maxW="xs">
              <FormLabel>Core Configuration</FormLabel>
              <Select value={selectedCoreConfigId} onChange={(e) => setSelectedCoreConfigId(e.target.value)}>
                {coreConfigs.length === 0 && (<option value="">System Defaults</option>)}
                {coreConfigs.map(c => (<option key={c.id} value={c.id}>{c.name}</option>))}
              </Select>
            </FormControl>
            <FormControl maxW="xs">
              <FormLabel>Game</FormLabel>
              {games.length > 0 ? (
                <Select value={selectedGameId} onChange={(e) => setSelectedGameId(e.target.value)}>
                  {games.map(g => (<option key={g.id} value={g.id}>{g.name}</option>))}
                </Select>
              ) : (
                <HStack>
                  <Text color="gray.500">No games found.</Text>
                  <Button leftIcon={<AddIcon />} onClick={quickCreateGame} variant="outline" className="db-btn">Create Game</Button>
                </HStack>
              )}
            </FormControl>
          </HStack>
        </CardHeader>
        <CardBody>
          <Grid templateColumns={{ base: '1fr', md: '1fr 1fr' }} gap={6}>
            <Box>
              <Heading size="sm" mb={3}>Add Player</Heading>
              <VStack align="stretch" spacing={3}>
                <FormControl>
                  <FormLabel>Role</FormLabel>
                  <Select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
                    {roleOptions.map(o => (<option key={o.value} value={o.value}>{o.label}</option>))}
                  </Select>
                </FormControl>
                <FormControl>
                  <FormLabel>Type</FormLabel>
                  <Select value={form.type} onChange={(e) => setForm({ ...form, type: e.target.value })}>
                    <option value="agent">Agent</option>
                    <option value="human">Human</option>
                  </Select>
                </FormControl>
                {form.type === 'agent' && (
                  <FormControl>
                    <FormLabel>Agent Type</FormLabel>
                    <Select value={form.agent_type} onChange={(e) => setForm({ ...form, agent_type: e.target.value })}>
                      <option value="DAYBREAK_DTCE">Daybreak Agent - DTCE</option>
                      <option value="DAYBREAK_DTCE_CENTRAL">Daybreak Agent - DTCE + Central Override</option>
                      <option value="LLM_BALANCED">LLM - Balanced</option>
                      <option value="LLM_CONSERVATIVE">LLM - Conservative</option>
                      <option value="LLM_AGGRESSIVE">LLM - Aggressive</option>
                      <option value="NAIVE">Heuristic - Naive</option>
                      <option value="BULLWHIP">Heuristic - Bullwhip</option>
                    </Select>
                  </FormControl>
                )}
                {form.type === 'human' && (
                  <FormControl>
                    <FormLabel>User</FormLabel>
                    <Select placeholder="Select user" value={form.user_id} onChange={(e) => setForm({ ...form, user_id: e.target.value })}>
                      {users.map(u => (<option key={u.id} value={u.id}>{u.email || u.username}</option>))}
                    </Select>
                  </FormControl>
                )}
                <FormControl>
                  <FormLabel>Display Name (optional)</FormLabel>
                  <Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="e.g., Retailer (Trevor)" />
                </FormControl>
                <Button colorScheme="blue" onClick={handleAdd}>Add Player</Button>
              </VStack>
            </Box>
            <Box>
              <Heading size="sm" mb={3}>Current Players</Heading>
              {players.length === 0 ? (
                <Text color="gray.500">No players yet for this game.</Text>
              ) : (
                <VStack align="stretch" spacing={2}>
                  {players.map(p => (
                    <Box key={p.id} borderWidth="1px" borderRadius="md" p={3}>
                      <HStack justify="space-between">
                        <Text fontWeight="600">{p.name}</Text>
                        <Text textTransform="capitalize" color="gray.600">{p.role.toLowerCase()}</Text>
                        <Text color="gray.600">{p.player_type === 'agent' || p.is_ai ? 'AI' : 'Human'}</Text>
                        {p.user_id && <Text color="gray.500">User #{p.user_id}</Text>}
                      </HStack>
                    </Box>
                  ))}
                </VStack>
              )}
            </Box>
          </Grid>
        </CardBody>
      </Card>
    </PageLayout>
  );
};

export default PlayersPage;
