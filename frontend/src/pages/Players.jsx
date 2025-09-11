import React, { useEffect, useMemo, useState } from 'react';
import { Box, Button, Card, CardBody, CardHeader, FormControl, FormLabel, Grid, Heading, HStack, Input, Select, Spinner, Text, useToast, VStack } from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import PageLayout from '../components/PageLayout';
import mixedGameApi, { api } from '../services/api';

const roleOptions = [
  { value: 'retailer', label: 'Retailer' },
  { value: 'wholesaler', label: 'Wholesaler' },
  { value: 'distributor', label: 'Distributor' },
  { value: 'factory', label: 'Factory' },
];

const PlayersPage = () => {
  const toast = useToast();
  const [games, setGames] = useState([]);
  const [users, setUsers] = useState([]);
  const [selectedGameId, setSelectedGameId] = useState('');
  const [loading, setLoading] = useState(true);
  const [players, setPlayers] = useState([]);

  const [form, setForm] = useState({
    name: '',
    role: 'retailer',
    type: 'human',
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
        const [gamesData, usersRes] = await Promise.all([
          mixedGameApi.getGames(),
          api.get('/auth/users/'),
        ]);
        setGames(gamesData);
        setUsers(usersRes.data || []);
        if (gamesData?.length) {
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
        name: form.name || `${form.role} (${form.type === 'human' ? 'Human' : 'AI'})`,
        role: form.role,
        is_ai: form.type !== 'human',
        user_id: form.type === 'human' ? Number(form.user_id) || null : null,
      };
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
              <FormLabel>Game</FormLabel>
              <Select value={selectedGameId} onChange={(e) => setSelectedGameId(e.target.value)}>
                {games.map(g => (<option key={g.id} value={g.id}>{g.name}</option>))}
              </Select>
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
                    <option value="human">Human</option>
                    <option value="ai">AI</option>
                  </Select>
                </FormControl>
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

