import React, { useState, useEffect } from 'react';
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
  Icon,
  Flex,
  Card,
  CardHeader,
  CardBody
} from '@chakra-ui/react';
import { FiPlus } from 'react-icons/fi';
import PageLayout from '../components/PageLayout';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import FilterBar from '../components/FilterBar';
import KPIStat from '../components/KPIStat';
import SkuTable from '../components/SkuTable';

// Sample data for the template-like dashboard
const demandSeries = [
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

const stockVsSafety = [
  { name: 'Widget A', stock: 1800, safety: 400 },
  { name: 'Widget B', stock: 900, safety: 300 },
  { name: 'Component C', stock: 7500, safety: 800 },
  { name: 'Assembly D', stock: 1200, safety: 350 },
  { name: 'Module E', stock: 320, safety: 100 },
  { name: 'Part F', stock: 6780, safety: 500 },
];

const stockVsForecast = [
  { name: 'Widget A', stock: 1800, forecast: 2200 },
  { name: 'Widget B', stock: 900, forecast: 1200 },
  { name: 'Component C', stock: 7500, forecast: 3900 },
  { name: 'Assembly D', stock: 1200, forecast: 1500 },
  { name: 'Module E', stock: 320, forecast: 500 },
  { name: 'Part F', stock: 6780, forecast: 5200 },
];

// function useQuery() {
//   const { search } = useLocation();
//   return new URLSearchParams(search);
// }

const Dashboard = () => {
  // Theme values - must be called unconditionally at the top level
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  // State
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  // const query = useQuery();
  // Query params used for future deep-linking
  // const gameId = query.get('gameId');


  useEffect(() => {
    // Simulate initial dashboard data load; route is already protected
    setLoading(true);
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minH="50vh">
        <CircularProgress isIndeterminate color="blue.500" />
      </Box>
    );
  }

  return (
    <PageLayout title="Dashboard">
      <Box p={4}>
        <Flex justify="space-between" align="center" mb={6} mt={2}>
          <VStack align="flex-start" spacing={1}>
            <Heading size="xl" fontWeight="600">Dashboard</Heading>
            <Text color="gray.500" fontSize="md">
              Overview of your supply chain performance
            </Text>
          </VStack>
        </Flex>
        <FilterBar />
        
        {/* KPI Cards */}
        <Grid templateColumns={{ base: '1fr', sm: 'repeat(2, 1fr)', lg: 'repeat(4, 1fr)' }} gap={4} mb={6}>
          <GridItem>
            <KPIStat 
              title="Total Inventory Value" 
              value="$1,245,678" 
              change={+2.3} 
              icon="inventory"
            />
          </GridItem>
          <GridItem>
            <KPIStat 
              title="Stockouts (Last 30d)" 
              value="12" 
              change={-15.4} 
              icon="warning"
              trend="down"
            />
          </GridItem>
          <GridItem>
            <KPIStat 
              title="On-Time Delivery" 
              value="94.5%" 
              change={1.2} 
              icon="delivery"
            />
          </GridItem>
          <GridItem>
            <KPIStat 
              title="Forecast Accuracy" 
              value="88.2%" 
              change={0.8} 
              icon="forecast"
            />
          </GridItem>
        </Grid>

        {/* Main Content Grid */}
        <Grid templateColumns={{ base: '1fr', lg: '2fr 1fr' }} gap={6} mb={6}>
          {/* Left Column */}
          <GridItem>
            {/* Demand Chart */}
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
                      <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" strokeWidth={2} />
                      <Line type="monotone" dataKey="forecast" stroke="#82ca9d" name="Forecast" strokeWidth={2} strokeDasharray="5 5" />
                      <Line type="monotone" dataKey="target" stroke="#ff7300" name="Target" strokeWidth={1} strokeDasharray="3 3" />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>

            {/* Stock vs Forecast */}
            <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6} className="card-surface pad-6">
              <CardHeader display="flex" justifyContent="space-between" alignItems="center">
                <Heading size="md">Stock vs Forecast (Next 4 Weeks)</Heading>
                <Button 
                  leftIcon={<Icon as={FiPlus} />}
                  bg="blue.600"
                  color="white"
                  _hover={{
                    bg: 'blue.700',
                    transform: 'translateY(-1px)',
                    boxShadow: 'md',
                  }}
                  _active={{
                    bg: 'blue.800',
                    transform: 'translateY(0)',
                  }}
                  onClick={() => navigate('/create-forecast')}
                >
                  New Forecast
                </Button>
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
                      <Bar dataKey="stock" fill="#8884d8" name="Current Stock" />
                      <Bar dataKey="forecast" fill="#ffc658" name="4-Week Forecast" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          </GridItem>
          {/* Right Column */}
          <GridItem>
            {/* Stock vs Safety Stock */}
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
                      <Bar dataKey="stock" fill="#8884d8" name="Current Stock" />
                      <Bar dataKey="safety" fill="#82ca9d" name="Safety Stock" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          </GridItem>
        </Grid>

        {/* Full Width Section */}
        <Card variant="outline" bg={cardBg} borderColor={borderColor} className="card-surface">
          <CardBody p={0} className="pad-6">
            <SkuTable data={[]} />
          </CardBody>
        </Card>
      </Box>
    </PageLayout>
  );
};

export default Dashboard;
