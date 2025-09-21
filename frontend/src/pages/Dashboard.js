import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Grid,
  GridItem,
  Button,
  CircularProgress,
  Text,
  Heading,
  useColorModeValue,
  VStack,
  HStack,
  Flex,
  Card,
  CardHeader,
  CardBody,
  Badge,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useToast,
  Alert,
  AlertIcon,
} from '@chakra-ui/react';
import PageLayout from '../components/PageLayout';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import FilterBar from '../components/FilterBar';
import KPIStat from '../components/KPIStat';
import SkuTable from '../components/SkuTable';
import { useAuth } from '../contexts/AuthContext';
import { getHumanDashboard } from '../services/dashboardService';

const FALLBACK_DEMAND_SERIES = [
  { name: 'W1', actual: 2100, forecast: 2200, target: 2000 },
  { name: 'W2', actual: 2250, forecast: 2300, target: 2050 },
  { name: 'W3', actual: 2150, forecast: 2350, target: 2100 },
  { name: 'W4', actual: 2300, forecast: 2400, target: 2100 },
  { name: 'W5', actual: 2400, forecast: 2380, target: 2150 },
  { name: 'W6', actual: 2350, forecast: 2450, target: 2150 },
  { name: 'W7', actual: 2420, forecast: 2500, target: 2200 },
  { name: 'W8', actual: 2380, forecast: 2480, target: 2200 },
  { name: 'W9', actual: 2450, forecast: 2550, target: 2250 },
  { name: 'W10', actual: 2480, forecast: 2580, target: 2250 },
  { name: 'W11', actual: 2460, forecast: 2600, target: 2300 },
  { name: 'W12', actual: 2520, forecast: 2650, target: 2300 },
];

const FALLBACK_STOCK_VS_SAFETY = [
  { name: 'Widget A', stock: 1800, safety: 400 },
  { name: 'Widget B', stock: 900, safety: 300 },
  { name: 'Component C', stock: 7500, safety: 800 },
  { name: 'Assembly D', stock: 1200, safety: 350 },
  { name: 'Module E', stock: 320, safety: 100 },
  { name: 'Part F', stock: 6780, safety: 500 },
];

const FALLBACK_STOCK_VS_FORECAST = [
  { name: 'Widget A', stock: 1800, forecast: 2200 },
  { name: 'Widget B', stock: 900, forecast: 1200 },
  { name: 'Component C', stock: 7500, forecast: 3900 },
  { name: 'Assembly D', stock: 1200, forecast: 1500 },
  { name: 'Module E', stock: 320, forecast: 500 },
  { name: 'Part F', stock: 6780, forecast: 5200 },
];

const FALLBACK_TOTAL_ROUNDS = 36;
const FALLBACK_CURRENT_ROUND = 18;

const FALLBACK_DECISION_TIMELINE = [
  {
    week: 18,
    order: 2450,
    reason: 'Anticipating a regional promotion and aligning safety stock.',
    inventory: 1820,
    backlog: 90,
  },
  {
    week: 17,
    order: 2320,
    reason: 'Backlog declined after expedited shipment; stabilizing pipeline.',
    inventory: 1765,
    backlog: 110,
  },
  {
    week: 16,
    order: 2280,
    reason: 'Maintained previous order to absorb variability from manufacturer delay.',
    inventory: 1690,
    backlog: 140,
  },
  {
    week: 15,
    order: 2200,
    reason: 'Raised order size to offset three-week moving average increase.',
    inventory: 1580,
    backlog: 180,
  },
];

const normalizeNumber = (value) => {
  if (value === null || value === undefined) {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const formatNumber = (value, options) => {
  const numeric = normalizeNumber(value);
  if (numeric === null) {
    return '--';
  }
  return new Intl.NumberFormat('en-US', options).format(numeric);
};

const formatCurrency = (value) => formatNumber(value, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

const formatPercent = (value) => {
  const numeric = normalizeNumber(value);
  if (numeric === null) {
    return '--';
  }
  return `${(numeric * 100).toFixed(1)}%`;
};

const formatSigned = (value, { asPercent = false } = {}) => {
  const numeric = normalizeNumber(value);
  if (numeric === null) {
    return null;
  }
  const prefix = numeric > 0 ? '+' : '';
  if (asPercent) {
    return `${prefix}${(numeric * 100).toFixed(1)}%`;
  }
  return `${prefix}${formatNumber(numeric)}`;
};

const Dashboard = () => {
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const timelineBg = useColorModeValue('gray.50', 'gray.700');
  const navigate = useNavigate();
  const toast = useToast();
  const { logout, user } = useAuth();

  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [assignmentModalOpen, setAssignmentModalOpen] = useState(false);
  const [assignmentMessage, setAssignmentMessage] = useState('You are not assigned to a game yet. Please contact your facilitator to be added to a session.');
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    const fetchDashboard = async () => {
      setLoading(true);
      try {
        const data = await getHumanDashboard();
        if (!isMounted) {
          return;
        }
        setDashboardData(data);
        setError(null);
        setAssignmentModalOpen(false);
      } catch (err) {
        if (!isMounted) {
          return;
        }
        const status = err?.response?.status;
        if (status === 404) {
          setAssignmentMessage(err?.response?.data?.detail || 'We could not find an active game assignment for your account.');
          setDashboardData(null);
          setAssignmentModalOpen(true);
        } else if (status === 403) {
          setAssignmentMessage('You do not have access to the player dashboard.');
          setDashboardData(null);
          setAssignmentModalOpen(true);
        } else {
          console.error('Unable to load dashboard data:', err);
          setError('Unable to load the dashboard right now. Please try again later.');
          toast({
            title: 'Failed to load dashboard',
            description: err?.response?.data?.detail || err?.message || 'Unexpected error occurred.',
            status: 'error',
            duration: 9000,
            isClosable: true,
          });
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchDashboard();

    return () => {
      isMounted = false;
    };
  }, [toast, user?.id]);

  const handleAssignmentLogout = async () => {
    try {
      await logout();
    } finally {
      navigate('/login');
    }
  };

  const metrics = dashboardData?.metrics;
  const timeSeries = dashboardData?.time_series ?? [];
  const totalRounds = normalizeNumber(dashboardData?.max_rounds) || FALLBACK_TOTAL_ROUNDS;
  const currentRoundRaw = normalizeNumber(dashboardData?.current_round) ?? FALLBACK_CURRENT_ROUND;
  const sliderMax = totalRounds > 0 ? totalRounds : FALLBACK_TOTAL_ROUNDS;
  const sliderValue = Math.min(Math.max(currentRoundRaw, 0), sliderMax);
  const progressPercent = sliderMax ? Math.round((sliderValue / sliderMax) * 100) : 0;
  const playerRoleLabel = dashboardData?.player_role ? dashboardData.player_role.replace(/_/g, ' ') : null;

  const demandSeries = useMemo(() => {
    if (!timeSeries.length) {
      return FALLBACK_DEMAND_SERIES;
    }
    return timeSeries.map((point) => ({
      name: `W${point.week}`,
      actual: normalizeNumber(point.demand) ?? normalizeNumber(point.order) ?? 0,
      forecast: normalizeNumber(point.order) ?? 0,
      target: normalizeNumber(point.inventory) ?? 0,
    }));
  }, [timeSeries]);

  const stockVsSafety = useMemo(() => {
    if (!timeSeries.length) {
      return FALLBACK_STOCK_VS_SAFETY;
    }
    return timeSeries.slice(-6).map((point) => ({
      name: `Week ${point.week}`,
      stock: normalizeNumber(point.inventory) ?? 0,
      safety: normalizeNumber(point.inventory) !== null && normalizeNumber(point.backlog) !== null
        ? Math.max(0, normalizeNumber(point.inventory) - normalizeNumber(point.backlog))
        : normalizeNumber(point.backlog) ?? 0,
    }));
  }, [timeSeries]);

  const stockVsForecast = useMemo(() => {
    if (!timeSeries.length) {
      return FALLBACK_STOCK_VS_FORECAST;
    }
    return timeSeries.slice(-6).map((point) => ({
      name: `Week ${point.week}`,
      stock: normalizeNumber(point.inventory) ?? 0,
      forecast: normalizeNumber(point.order) ?? 0,
    }));
  }, [timeSeries]);

  const decisionTimeline = useMemo(() => {
    if (!timeSeries.length) {
      return FALLBACK_DECISION_TIMELINE;
    }
    return [...timeSeries]
      .slice(-6)
      .reverse()
      .map((point) => ({
        week: point.week,
        order: normalizeNumber(point.order) ?? 0,
        inventory: normalizeNumber(point.inventory) ?? 0,
        backlog: normalizeNumber(point.backlog) ?? 0,
        reason: point.reason || 'No decision notes captured.',
      }));
  }, [timeSeries]);

  const kpiCards = useMemo(() => {
    if (!metrics) {
      return [
        { title: 'Current Inventory', value: '--', subtitle: 'Units on hand' },
        { title: 'Backlog', value: '--', subtitle: 'Open orders' },
        { title: 'Average Weekly Cost', value: '--', subtitle: 'Recent average' },
        { title: 'Service Level', value: '--', subtitle: 'Fulfillment rate' },
      ];
    }

    const inventoryChange = formatSigned(metrics.inventory_change);
    const serviceLevelChange = formatSigned(metrics.service_level_change, { asPercent: true });

    return [
      {
        title: 'Current Inventory',
        value: formatNumber(metrics.current_inventory),
        subtitle: 'Units on hand',
        delta: inventoryChange,
        deltaPositive: (normalizeNumber(metrics.inventory_change) ?? 0) >= 0,
      },
      {
        title: 'Backlog',
        value: formatNumber(metrics.backlog),
        subtitle: 'Outstanding demand',
        delta: null,
      },
      {
        title: 'Average Weekly Cost',
        value: formatCurrency(metrics.avg_weekly_cost),
        subtitle: 'Rolling average',
        delta: null,
      },
      {
        title: 'Service Level',
        value: formatPercent(metrics.service_level),
        subtitle: 'Fulfillment rate',
        delta: serviceLevelChange,
        deltaPositive: (normalizeNumber(metrics.service_level_change) ?? 0) >= 0,
      },
    ];
  }, [metrics]);

  const pageTitle = dashboardData?.game_name ? `${dashboardData.game_name} Dashboard` : 'Dashboard';

  return (
    <>
      <PageLayout title={pageTitle}>
        <Box p={4}>
          {loading ? (
            <Box display="flex" justifyContent="center" alignItems="center" minH="50vh">
              <CircularProgress isIndeterminate color="blue.500" />
            </Box>
          ) : (
            <>
              {error && (
                <Alert status="error" mb={4}>
                  <AlertIcon />
                  {error}
                </Alert>
              )}

              <Flex justify="space-between" align={{ base: 'flex-start', md: 'center' }} mb={6} mt={2} direction={{ base: 'column', md: 'row' }} gap={4}>
                <VStack align="flex-start" spacing={1}>
                  <Heading size="xl" fontWeight="600">{dashboardData?.game_name || 'Dashboard'}</Heading>
                  <Text color="gray.500" fontSize="md">Overview of your supply chain performance</Text>
                  <HStack spacing={2} mt={1}>
                    {playerRoleLabel && (
                      <Badge colorScheme="blue" textTransform="capitalize">{playerRoleLabel}</Badge>
                    )}
                    <Badge colorScheme="purple">Round {sliderValue} / {sliderMax}</Badge>
                  </HStack>
                </VStack>
              </Flex>

              <FilterBar />

              <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6} className="card-surface pad-6">
                <CardHeader pb={2}>
                  <Heading size="md">Game Progress</Heading>
                </CardHeader>
                <CardBody pt={0}>
                  <VStack align="stretch" spacing={3}>
                    <Flex justify="space-between" align="center">
                      <Text color="gray.500">Rounds completed</Text>
                      <Text fontWeight="semibold">{progressPercent}%</Text>
                    </Flex>
                    <Slider value={sliderValue} min={0} max={sliderMax} isReadOnly focusThumbOnChange={false}>
                      <SliderTrack>
                        <SliderFilledTrack />
                      </SliderTrack>
                      <SliderThumb boxSize={5} fontSize="xs">
                        <Text fontSize="xs">{sliderValue}</Text>
                      </SliderThumb>
                    </Slider>
                    <Text fontSize="sm" color="gray.500">Week {sliderValue} of {sliderMax}</Text>
                  </VStack>
                </CardBody>
              </Card>

              <Grid templateColumns={{ base: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }} gap={4} mb={6}>
                {kpiCards.map((card) => (
                  <GridItem key={card.title}>
                    <KPIStat
                      title={card.title}
                      value={card.value}
                      subtitle={card.subtitle}
                      delta={card.delta}
                      deltaPositive={card.deltaPositive}
                    />
                  </GridItem>
                ))}
              </Grid>

              <Grid templateColumns={{ base: '1fr', lg: '2fr 1fr' }} gap={6} mb={6}>
                <GridItem>
                  <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6} className="card-surface pad-6">
                    <CardHeader>
                      <Heading size="md">Demand Forecast vs Actual</Heading>
                    </CardHeader>
                    <CardBody>
                      <Box h="300px">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={demandSeries}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="actual" stroke="#3182CE" name="Actual" strokeWidth={2} />
                            <Line type="monotone" dataKey="forecast" stroke="#38A169" name="Forecast" strokeWidth={2} strokeDasharray="5 5" />
                            <Line type="monotone" dataKey="target" stroke="#DD6B20" name="Target" strokeWidth={1} strokeDasharray="3 3" />
                          </LineChart>
                        </ResponsiveContainer>
                      </Box>
                    </CardBody>
                  </Card>

                  <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6} className="card-surface pad-6">
                    <CardHeader>
                      <Heading size="md">Stock vs Forecast (Recent)</Heading>
                    </CardHeader>
                    <CardBody>
                      <Box h="300px">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={stockVsForecast}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="stock" fill="#3182CE" name="Current Stock" />
                            <Bar dataKey="forecast" fill="#DD6B20" name="Forecast" />
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                    </CardBody>
                  </Card>
                </GridItem>

                <GridItem>
                  <Card variant="outline" bg={cardBg} borderColor={borderColor} h="100%" mb={6} className="card-surface pad-6">
                    <CardHeader>
                      <Heading size="md">Stock vs Safety Stock</Heading>
                    </CardHeader>
                    <CardBody>
                      <Box h="300px">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={stockVsSafety}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="stock" fill="#3182CE" name="Current Stock" />
                            <Bar dataKey="safety" fill="#38A169" name="Safety Stock" />
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                    </CardBody>
                  </Card>
                </GridItem>
              </Grid>

              <Card variant="outline" bg={cardBg} borderColor={borderColor} className="card-surface">
                <CardBody p={0} className="pad-6">
                  <SkuTable data={[]} />
                </CardBody>
              </Card>

              <Card variant="outline" bg={cardBg} borderColor={borderColor} className="card-surface" mt={6}>
                <CardHeader pb={2}>
                  <Heading size="md">Order Reasoning Timeline</Heading>
                  <Text color="gray.500" fontSize="sm">Most recent decisions are shown first</Text>
                </CardHeader>
                <CardBody pt={0}>
                  <VStack align="stretch" spacing={4}>
                    {decisionTimeline.map((entry) => (
                      <Box
                        key={`timeline-${entry.week}`}
                        p={4}
                        borderWidth="1px"
                        borderRadius="md"
                        borderColor={borderColor}
                        bg={timelineBg}
                      >
                        <Flex justify="space-between" align="flex-start" mb={2} gap={3} flexWrap="wrap">
                          <HStack spacing={3} align="center">
                            <Badge colorScheme="blue">Week {entry.week}</Badge>
                            <Badge colorScheme="purple" variant="subtle">Order {entry.order}</Badge>
                          </HStack>
                          <Text fontSize="sm" color="gray.500">Inventory {entry.inventory} · Backlog {entry.backlog}</Text>
                        </Flex>
                        <Text fontSize="sm" color="gray.700">{entry.reason}</Text>
                      </Box>
                    ))}
                  </VStack>
                </CardBody>
              </Card>
            </>
          )}
        </Box>
      </PageLayout>

      <Modal isOpen={assignmentModalOpen} onClose={() => {}} isCentered closeOnOverlayClick={false} closeOnEsc={false}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Join a Game</ModalHeader>
          <ModalBody>
            <Text mb={4}>{assignmentMessage}</Text>
            <Text fontSize="sm" color="gray.500">
              Once you have been assigned to a game, log in again to access the dashboard.
            </Text>
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" onClick={handleAssignmentLogout}>
              Return to Login
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
};

export default Dashboard;
