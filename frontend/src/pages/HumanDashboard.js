import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Box,
  Heading,
  Text,
  VStack,
  HStack,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  useColorModeValue,
  Spinner,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Badge,
  Divider,
  useToast,
  FormControl,
  FormLabel,
  NumberInput,
  NumberInputField,
  Textarea,
  Button,
  FormErrorMessage,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from '@chakra-ui/react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { format } from 'date-fns';
import { useAuth } from '../contexts/AuthContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import { getHumanDashboard, formatChartData } from '../services/dashboardService';
import mixedGameApi from '../services/api';
import PageLayout from '../components/PageLayout';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ROLE_COLORS = {
  RETAILER: 'blue',
  WHOLESALER: 'green',
  DISTRIBUTOR: 'purple',
  MANUFACTURER: 'orange',
  SUPPLIER: 'red'
};

const HumanDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [orderQuantity, setOrderQuantity] = useState('');
  const [orderReason, setOrderReason] = useState('');
  const [orderError, setOrderError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { user } = useAuth();
  const { connect, subscribe } = useWebSocket();
  const toast = useToast();
  const lastRoundRef = useRef(null);

  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const timelineBg = useColorModeValue('gray.50', 'gray.700');

  const fetchDashboardData = useCallback(
    async (withLoader = false) => {
      try {
        if (withLoader) {
          setLoading(true);
        }
        setError(null);
        const data = await getHumanDashboard();
        setDashboardData(data);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
        if (withLoader) {
          toast({
            title: 'Error',
            description: 'Failed to load dashboard data. Please try again later.',
            status: 'error',
            duration: 5000,
            isClosable: true,
          });
        }
      } finally {
        if (withLoader) {
          setLoading(false);
        }
      }
    },
    [toast]
  );

  useEffect(() => {
    fetchDashboardData(true);
  }, [fetchDashboardData]);

  useEffect(() => {
    if (!dashboardData?.game_id || !dashboardData?.player_id) {
      return undefined;
    }

    connect(dashboardData.game_id, dashboardData.player_id);

    const unsubscribe = subscribe((event, payload) => {
      if (event !== 'message') {
        return;
      }

      const messageType = payload?.type;
      if (
        [
          'game_state',
          'round_completed',
          'round_started',
          'order_submitted',
          'inventory_update',
        ].includes(messageType)
      ) {
        fetchDashboardData(false);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [dashboardData?.game_id, dashboardData?.player_id, connect, subscribe, fetchDashboardData]);

  useEffect(() => {
    if (!dashboardData?.current_round) {
      return;
    }

    const currentRound = dashboardData.current_round;
    if (lastRoundRef.current === currentRound) {
      return;
    }

    const series = dashboardData.time_series || [];
    const matchingEntry =
      series.find(entry => entry.week === currentRound) ||
      [...series].sort((a, b) => (a.week ?? 0) - (b.week ?? 0)).pop();

    if (matchingEntry && typeof matchingEntry.order === 'number') {
      setOrderQuantity(String(matchingEntry.order));
    } else {
      setOrderQuantity('');
    }

    setOrderReason('');
    setOrderError('');
    lastRoundRef.current = currentRound;
  }, [dashboardData, lastRoundRef]);

  const handleOrderSubmit = useCallback(async (event) => {
    event.preventDefault();
    setOrderError('');

    if (!dashboardData?.game_id || !dashboardData?.player_id) {
      setOrderError('Unable to determine the current game or player.');
      return;
    }

    if (orderQuantity === '') {
      setOrderError('Please enter an order quantity.');
      return;
    }

    const quantityValue = Number(orderQuantity);
    if (Number.isNaN(quantityValue) || quantityValue < 0) {
      setOrderError('Order quantity must be zero or a positive number.');
      return;
    }

    try {
      setIsSubmitting(true);
      await mixedGameApi.submitOrder(
        dashboardData.game_id,
        dashboardData.player_id,
        quantityValue,
        orderReason.trim() ? orderReason.trim() : undefined
      );

      toast({
        title: 'Order submitted',
        description: `Your order of ${quantityValue} units has been recorded.`,
        status: 'success',
        duration: 4000,
        isClosable: true,
      });

      await fetchDashboardData(false);
    } catch (err) {
      console.error('Failed to submit order:', err);
      const detail = err?.response?.data?.detail;
      const errorMessage = typeof detail === 'string' ? detail : 'Failed to submit order.';
      setOrderError(errorMessage);
      toast({
        title: 'Order submission failed',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [dashboardData, orderQuantity, orderReason, toast, fetchDashboardData]);

  const sliderMax = useMemo(() => {
    if (!dashboardData) {
      return 0;
    }
    return Math.max(dashboardData.max_rounds || 0, dashboardData.current_round || 0);
  }, [dashboardData]);

  const sliderDisplayMax = sliderMax || 1;
  const sliderValue = Math.min(dashboardData?.current_round || 0, sliderDisplayMax);
  const progressPercent = sliderDisplayMax
    ? Math.round((sliderValue / sliderDisplayMax) * 100)
    : 0;

  const reasoningTimeline = useMemo(() => {
    if (!dashboardData?.time_series?.length) {
      return [];
    }

    return [...dashboardData.time_series]
      .sort((a, b) => (b.week ?? 0) - (a.week ?? 0));
  }, [dashboardData]);

  const renderMetrics = () => {
    if (!dashboardData?.metrics) return null;

    const { metrics } = dashboardData;
    const serviceLevelPercent = (metrics.service_level || 0) * 100;
    const serviceLevelChangePercent = (metrics.service_level_change || 0) * 100;
    
    return (
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4} mb={6}>
        <Stat p={4} bg={cardBg} borderRadius="lg" boxShadow="sm" borderWidth="1px" borderColor={borderColor}>
          <StatLabel>Current Inventory</StatLabel>
          <StatNumber>{Math.round(metrics.current_inventory || 0)}</StatNumber>
          <StatHelpText>
            <StatArrow type={metrics.inventory_change >= 0 ? 'increase' : 'decrease'} />
            {Math.abs(metrics.inventory_change || 0).toFixed(1)}% from last week
          </StatHelpText>
        </Stat>

        <Stat p={4} bg={cardBg} borderRadius="lg" boxShadow="sm" borderWidth="1px" borderColor={borderColor}>
          <StatLabel>Current Backlog</StatLabel>
          <StatNumber color={metrics.backlog > 0 ? 'red.500' : 'inherit'}>
            {Math.round(metrics.backlog || 0)}
          </StatNumber>
          <StatHelpText>
            {metrics.backlog > 0 ? 'Orders pending' : 'No pending orders'}
          </StatHelpText>
        </Stat>

        <Stat p={4} bg={cardBg} borderRadius="lg" boxShadow="sm" borderWidth="1px" borderColor={borderColor}>
          <StatLabel>Total Cost</StatLabel>
          <StatNumber>${(metrics.total_cost || 0).toFixed(2)}</StatNumber>
          <StatHelpText>${(metrics.avg_weekly_cost || 0).toFixed(2)} per week</StatHelpText>
        </Stat>

        <Stat p={4} bg={cardBg} borderRadius="lg" boxShadow="sm" borderWidth="1px" borderColor={borderColor}>
          <StatLabel>Service Level</StatLabel>
          <StatNumber>{serviceLevelPercent.toFixed(1)}%</StatNumber>
          <StatHelpText>
            <StatArrow type={serviceLevelChangePercent >= 0 ? 'increase' : 'decrease'} />
            {Math.abs(serviceLevelChangePercent).toFixed(1)}% from last week
          </StatHelpText>
        </Stat>
      </SimpleGrid>
    );
  };

  const renderChart = () => {
    if (!dashboardData?.time_series?.length) return null;

    const { time_series, player_role } = dashboardData;
    const chartData = formatChartData(time_series, player_role);
    const labels = chartData.map(item => `Week ${item.week}`);
    
    const datasets = [
      {
        label: 'Inventory',
        data: chartData.map(item => item.inventory || 0),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.3,
        fill: true,
        yAxisID: 'y',
      },
      {
        label: 'Orders',
        data: chartData.map(item => item.order || 0),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.3,
        fill: false,
        yAxisID: 'y',
      },
      {
        label: 'Backlog',
        data: chartData.map(item => item.backlog || 0),
        borderColor: 'rgb(255, 159, 64)',
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        tension: 0.3,
        fill: false,
        yAxisID: 'y',
      },
      {
        label: 'Cost',
        data: chartData.map(item => item.cost || 0),
        borderColor: 'rgb(201, 203, 207)',
        backgroundColor: 'rgba(201, 203, 207, 0.2)',
        tension: 0.3,
        fill: false,
        yAxisID: 'y1',
        hidden: true, // Hidden by default as it might be on a different scale
      }
    ];

    // Add demand/supply based on role
    const showDemand = player_role === 'RETAILER' || player_role === 'MANUFACTURER' || player_role === 'DISTRIBUTOR';
    const showSupply = player_role === 'SUPPLIER' || player_role === 'MANUFACTURER' || player_role === 'DISTRIBUTOR';

    if (showDemand) {
      datasets.push({
        label: 'Demand',
        data: chartData.map(item => item.demand || 0),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.3,
        fill: false,
        borderDash: [5, 5],
        yAxisID: 'y',
      });
    }

    if (showSupply) {
      datasets.push({
        label: 'Supply',
        data: chartData.map(item => item.supply || 0),
        borderColor: 'rgb(153, 102, 255)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        tension: 0.3,
        fill: false,
        borderDash: [5, 5],
        yAxisID: 'y',
      });
    }

    const options = {
      responsive: true,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          position: 'top',
          onClick: (e, legendItem, legend) => {
            // Custom legend click handler to allow toggling individual datasets
            const index = legend.chart.data.datasets.findIndex(
              (ds) => ds.label === legendItem.text
            );
            if (index === -1) return;

            const meta = legend.chart.getDatasetMeta(index);
            meta.hidden = meta.hidden === null ? !legend.chart.data.datasets[index].hidden : null;
            legend.chart.update();
          },
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (context.parsed.y !== null) {
                label += Number.isInteger(context.parsed.y) 
                  ? context.parsed.y 
                  : context.parsed.y.toFixed(2);
              }
              return label;
            }
          }
        },
        title: {
          display: true,
          text: 'Weekly Performance Metrics',
        },
      },
      scales: {
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: {
            display: true,
            text: 'Units',
          },
          beginAtZero: true,
        },
        y1: {
          type: 'linear',
          display: false, // Hidden by default, can be toggled
          position: 'right',
          title: {
            display: true,
            text: 'Cost ($)',
          },
          beginAtZero: true,
          grid: {
            drawOnChartArea: false,
          },
        },
        x: {
          title: {
            display: true,
            text: 'Week'
          },
          grid: {
            display: false,
          },
        }
      },
    };

    return (
      <Box 
        p={4} 
        bg={cardBg} 
        borderRadius="lg" 
        boxShadow="sm" 
        borderWidth="1px" 
        borderColor={borderColor}
        mb={6}
      >
        <Line options={options} data={{ labels, datasets }} />
      </Box>
    );
  };

  if (loading) {
    return (
      <PageLayout title="Loading Dashboard...">
        <VStack spacing={4} mt={8}>
          <Spinner size="xl" />
          <Text>Loading your dashboard...</Text>
        </VStack>
      </PageLayout>
    );
  }

  if (error) {
    return (
      <PageLayout title="Error">
        <Alert status="error" mb={4}>
          <AlertIcon />
          <AlertTitle>Error loading dashboard</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </PageLayout>
    );
  }

  if (!dashboardData) {
    return (
      <PageLayout title="No Active Game">
        <Alert status="info" mb={4}>
          <AlertIcon />
          You are not currently part of any active game.
        </Alert>
      </PageLayout>
    );
  }

  const { game_name, player_role, current_round, last_updated } = dashboardData;
  const roleColor = ROLE_COLORS[player_role] || 'gray';

  return (
    <PageLayout title="My Dashboard">
      <VStack spacing={6} align="stretch">
        {/* Header */}
        <Box
          p={6}
          bg={cardBg}
          borderRadius="lg"
          boxShadow="sm"
          borderWidth="1px"
          borderColor={borderColor}
        >
          <HStack justify="space-between" mb={4} align="flex-start">
            <Box>
              <Heading size="lg">{game_name || 'My Game'}</Heading>
              <Text color="gray.500" mt={1}>
                Welcome back, {user?.username || 'Player'}
              </Text>
            </Box>
            <Box textAlign="right">
              <Badge
                colorScheme={roleColor.toLowerCase()}
                fontSize="1em"
                p={2}
                borderRadius="md"
              >
                {player_role || 'PLAYER'}
              </Badge>
              <Text mt={1} fontSize="sm" color="gray.500">
                Round {current_round || 1}
              </Text>
            </Box>
          </HStack>
          <Divider my={4} />
          <VStack align="stretch" spacing={3}>
            <Text fontSize="sm" color="gray.500">
              Last updated: {last_updated ? format(new Date(last_updated), 'PPpp') : '—'}
            </Text>
            <Box>
              <HStack justify="space-between" mb={2}>
                <Text fontSize="sm" fontWeight="medium">
                  Game Progress
                </Text>
                <Text fontSize="sm" color="gray.500">
                  {`${sliderValue} / ${sliderDisplayMax}`} ({progressPercent}% complete)
                </Text>
              </HStack>
              <Slider value={sliderValue} min={0} max={sliderDisplayMax} isReadOnly focusThumbOnChange={false}>
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb boxSize={5} fontSize="xs">
                  <Text fontSize="xs">{sliderValue}</Text>
                </SliderThumb>
              </Slider>
            </Box>
          </VStack>
        </Box>

        {/* Order submission */}
        <Box
          as="form"
          onSubmit={handleOrderSubmit}
          p={6}
          bg={cardBg}
          borderRadius="lg"
          boxShadow="sm"
          borderWidth="1px"
          borderColor={borderColor}
        >
          <Heading size="md" mb={4}>
            Submit order for Week {current_round || 1}
          </Heading>
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
            <FormControl isRequired isInvalid={Boolean(orderError)}>
              <FormLabel>Order quantity</FormLabel>
              <NumberInput
                min={0}
                value={orderQuantity}
                onChange={(valueString) => setOrderQuantity(valueString)}
                clampValueOnBlur={false}
                keepWithinRange={false}
              >
                <NumberInputField placeholder="Enter units to order" />
              </NumberInput>
              <FormErrorMessage>{orderError}</FormErrorMessage>
            </FormControl>
            <FormControl>
              <FormLabel>Reason (optional)</FormLabel>
              <Textarea
                value={orderReason}
                onChange={(event) => setOrderReason(event.target.value)}
                placeholder="Provide context for your order decision"
                rows={orderReason ? 4 : 3}
              />
            </FormControl>
          </SimpleGrid>
          <HStack justify="space-between" mt={4} flexWrap="wrap" spacing={3}>
            <Text fontSize="sm" color="gray.500">
              Round {current_round || 1} of {sliderDisplayMax}. Keep orders flowing to avoid backlog.
            </Text>
            <Button type="submit" colorScheme="blue" isLoading={isSubmitting} loadingText="Submitting">
              Submit order
            </Button>
          </HStack>
        </Box>

        {/* Metrics */}
        <Box>
          <Heading size="md" mb={4}>
            Performance Metrics
          </Heading>
          {renderMetrics()}
        </Box>

        {/* Chart */}
        <Box>
          <Heading size="md" mb={4}>
            Weekly Performance
          </Heading>
          {renderChart()}
        </Box>

        {/* Order reasoning timeline */}
        <Box>
          <Heading size="md" mb={4}>
            Order Reasoning by Week
          </Heading>
          <Box
            p={4}
            bg={cardBg}
            borderRadius="lg"
            boxShadow="sm"
            borderWidth="1px"
            borderColor={borderColor}
          >
            {reasoningTimeline.length ? (
              <VStack align="stretch" spacing={3}>
                {reasoningTimeline.map((entry) => (
                  <Box
                    key={`reason-${entry.week}`}
                    p={4}
                    borderRadius="md"
                    borderWidth="1px"
                    borderColor={borderColor}
                    bg={timelineBg}
                  >
                    <HStack justify="space-between" align="flex-start" spacing={3}>
                      <HStack spacing={3} align="center">
                        <Badge colorScheme="blue">Week {entry.week}</Badge>
                        <Badge colorScheme="purple" variant="subtle">
                          Order {Math.round(entry.order ?? 0)}
                        </Badge>
                      </HStack>
                      <Text fontSize="sm" color="gray.500">
                        {entry.reason ? 'Reason documented' : 'No reason provided'}
                      </Text>
                    </HStack>
                    <Text mt={3} fontSize="sm" color="gray.700">
                      {entry.reason || 'No reasoning provided for this order.'}
                    </Text>
                    <HStack spacing={4} mt={3} fontSize="xs" color="gray.500">
                      <Text>Inventory: {Math.round(entry.inventory ?? 0)}</Text>
                      <Text>Backlog: {Math.round(entry.backlog ?? 0)}</Text>
                    </HStack>
                  </Box>
                ))}
              </VStack>
            ) : (
              <Text color="gray.500">
                No order history is available yet. Your reasoning will appear here as you submit orders.
              </Text>
            )}
          </Box>
        </Box>
      </VStack>
    </PageLayout>
  );
};

export default HumanDashboard;
