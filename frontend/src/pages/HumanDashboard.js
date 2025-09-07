import React, { useState, useEffect } from 'react';
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
  Container,
  useToast
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
import { getHumanDashboard, formatChartData } from '../services/dashboardService';
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
  const { user } = useAuth();
  const toast = useToast();

  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const data = await getHumanDashboard();
        setDashboardData(data);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
        toast({
          title: 'Error',
          description: 'Failed to load dashboard data. Please try again later.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [toast]);

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
          <HStack justify="space-between" mb={4}>
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
          <Divider my={3} />
          <Text fontSize="sm" color="gray.500">
            Last updated: {format(new Date(last_updated), 'PPpp')}
          </Text>
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
      </VStack>
    </PageLayout>
  );
};

export default HumanDashboard;
