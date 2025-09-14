import React, { useState } from 'react';
import PageLayout from '../../components/PageLayout';
import { Box, Grid, FormControl, FormLabel, Input, Select, NumberInput, NumberInputField, Button, Text, useToast, Textarea, HStack } from '@chakra-ui/react';
import { mixedGameApi } from '../../services/api';

export default function Training() {
  const [serverHost, setServerHost] = useState('aiserver.local');
  const [source, setSource] = useState('sim');
  const [windowSize, setWindowSize] = useState(12);
  const [horizon, setHorizon] = useState(1);
  const [epochs, setEpochs] = useState(10);
  const [device, setDevice] = useState('cpu');
  const [dataPath, setDataPath] = useState('');
  const [stepsTable, setStepsTable] = useState('beer_game_steps');
  const [dbUrl, setDbUrl] = useState('');
  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  // Simulation settings
  const [useSimpy, setUseSimpy] = useState(true);
  const [simAlpha, setSimAlpha] = useState(0.3);
  const [simWipK, setSimWipK] = useState(1.0);
  const [ranges, setRanges] = useState({
    supply_leadtime: [0, 6], order_leadtime: [0, 6], init_inventory: [4, 60],
    holding_cost: [0.1, 2.0], backlog_cost: [0.2, 4.0], max_inbound_per_link: [50, 300], max_order: [50, 300]
  });
  // Generated dataset info (optional)
  const toast = useToast();

  const launch = async () => {
    try {
      const payload = { server_host: serverHost, source, window: windowSize, horizon, epochs, device, steps_table: stepsTable, db_url: dbUrl || undefined };
      const data = await mixedGameApi.trainModel(payload);
      setJob(data);
      toast({ title: 'Training started', description: data.note || 'Running locally', status: 'success', duration: 3000, isClosable: true });
    } catch (e) {
      toast({ title: 'Failed to start training', description: e?.response?.data?.detail || e.message, status: 'error', duration: 5000, isClosable: true });
    }
  };

  React.useEffect(() => {
    let timer;
    if (job?.job_id) {
      const poll = async () => {
        try {
          const st = await mixedGameApi.getJobStatus(job.job_id);
          setJobStatus(st);
          if (st.running) {
            timer = setTimeout(poll, 2000);
          }
        } catch (e) {
          // stop polling on error
        }
      };
      poll();
    }
    return () => { if (timer) clearTimeout(timer); };
  }, [job?.job_id]);

  const generate = async () => {
    try {
      const param_ranges = Object.fromEntries(Object.entries(ranges).map(([k, v]) => [k, [Number(v[0]), Number(v[1])]]));
      const data = await mixedGameApi.generateData({
        num_runs: 64,
        T: 64,
        window: windowSize,
        horizon,
        param_ranges,
        use_simpy: useSimpy,
        sim_alpha: simAlpha,
        sim_wip_k: simWipK,
      });
      setDataPath(data.path || '');
      toast({ title: 'Dataset generated', description: data.path, status: 'success', duration: 4000, isClosable: true });
    } catch (e) {
      setDataPath('');
      toast({ title: 'Failed to generate dataset', description: e?.response?.data?.detail || e.message, status: 'error', duration: 5000, isClosable: true });
    }
  };

  return (
    <PageLayout title="Daybreak Agent Training">
      <Box className="card-surface pad-6">
        <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={6}>
          <FormControl>
            <FormLabel>Server Host</FormLabel>
            <Input value={serverHost} onChange={(e) => setServerHost(e.target.value)} placeholder="aiserver.local" />
          </FormControl>
          <FormControl>
            <FormLabel>Source</FormLabel>
            <Select value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="sim">Simulator</option>
              <option value="db">Database</option>
            </Select>
          </FormControl>
          <FormControl>
            <FormLabel>Window</FormLabel>
            <NumberInput value={windowSize} onChange={(v)=> setWindowSize(parseInt(v)||1)} min={1} max={64}><NumberInputField /></NumberInput>
          </FormControl>
          <FormControl>
            <FormLabel>Horizon</FormLabel>
            <NumberInput value={horizon} onChange={(v)=> setHorizon(parseInt(v)||1)} min={1} max={8}><NumberInputField /></NumberInput>
          </FormControl>
          <FormControl>
            <FormLabel>Epochs</FormLabel>
            <NumberInput value={epochs} onChange={(v)=> setEpochs(parseInt(v)||1)} min={1} max={1000}><NumberInputField /></NumberInput>
          </FormControl>
          <FormControl>
            <FormLabel>Device</FormLabel>
            <Select value={device} onChange={(e) => setDevice(e.target.value)}>
              <option value="cpu">CPU</option>
              <option value="cuda">GPU</option>
            </Select>
          </FormControl>
          {source === 'db' && (
            <>
              <FormControl>
                <FormLabel>DB URL</FormLabel>
                <Input value={dbUrl} onChange={(e)=> setDbUrl(e.target.value)} placeholder="mysql+pymysql://user:pass@host/db" />
              </FormControl>
              <FormControl>
                <FormLabel>Steps Table</FormLabel>
                <Input value={stepsTable} onChange={(e)=> setStepsTable(e.target.value)} />
              </FormControl>
            </>
          )}
        </Grid>
      <Box mt={6}>
          <Text fontWeight="semibold" mb={2}>Simulation Settings</Text>
          <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap={4} mb={4}>
            <FormControl>
              <FormLabel>Use SimPy</FormLabel>
              <Select value={useSimpy ? '1' : '0'} onChange={(e)=> setUseSimpy(e.target.value === '1')}>
                <option value='1'>Yes (default)</option>
                <option value='0'>No (discrete simulator)</option>
              </Select>
            </FormControl>
            <FormControl>
              <FormLabel>Smoothing Alpha</FormLabel>
              <NumberInput min={0} max={1} step={0.05} value={simAlpha} onChange={(v)=> setSimAlpha(parseFloat(v)||0)}>
                <NumberInputField />
              </NumberInput>
            </FormControl>
            <FormControl>
              <FormLabel>WIP Gain</FormLabel>
              <NumberInput min={0} max={5} step={0.1} value={simWipK} onChange={(v)=> setSimWipK(parseFloat(v)||0)}>
                <NumberInputField />
              </NumberInput>
            </FormControl>
          </Grid>
          <Text fontWeight="semibold" mb={2}>Generate Dataset (Simulator)</Text>
          <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={4}>
            {Object.entries(ranges).map(([k,v]) => (
              <HStack key={k}>
                <Text w="44" textTransform="capitalize">{k.replaceAll('_',' ')}</Text>
                <Input w="24" value={v[0]} onChange={(e)=> setRanges(prev => ({...prev, [k]: [e.target.value, prev[k][1]]}))} />
                <Input w="24" value={v[1]} onChange={(e)=> setRanges(prev => ({...prev, [k]: [prev[k][0], e.target.value]}))} />
              </HStack>
            ))}
          </Grid>
          {!dataPath && (
            <Text fontSize="sm" color="gray.500" mb={2}>No Training Data</Text>
          )}
          <HStack mt={3} justify="flex-end">
            <Button variant="outline" onClick={generate}>Generate Data</Button>
            <Button colorScheme="green" onClick={launch} isDisabled={!dataPath}>Launch Training</Button>
            {job?.job_id && jobStatus?.running && (
              <Button colorScheme="red" onClick={async () => { try { await mixedGameApi.stopJob(job.job_id); } catch (e) {} }}>Stop</Button>
            )}
          </HStack>
        </Box>
        {job && (
          <Box mt={6}>
            <Text fontWeight="semibold">Job ID: {job.job_id}</Text>
            <Text>Log: {job.log}</Text>
            <Text>Command: <code>{job.cmd}</code></Text>
            {job.note && <Text color="orange.600">Note: {job.note}</Text>}
            {jobStatus && (
              <Box mt={3} className="card-surface pad-6">
                <Text>Running: {String(jobStatus.running)}</Text>
                <Text>PID: {jobStatus.pid || '-'}</Text>
                <Text mt={2} fontWeight="semibold">Log tail:</Text>
                <Textarea value={jobStatus.log_tail || ''} readOnly rows={10} />
              </Box>
            )}
          </Box>
        )}
      </Box>
    </PageLayout>
  );
}
