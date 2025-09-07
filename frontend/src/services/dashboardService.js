import api from './api';

/**
 * Get human dashboard data
 * @returns {Promise<Object>} Dashboard data including game info, metrics, and time series
 */
export const getHumanDashboard = async () => {
  try {
    const response = await api.get('/dashboard/human-dashboard');
    
    // Transform the response to match the expected format
    const formattedData = {
      ...response.data,
      metrics: {
        current_inventory: response.data.metrics.current_inventory,
        inventory_change: response.data.metrics.inventory_change,
        backlog: response.data.metrics.backlog,
        total_cost: response.data.metrics.total_cost,
        avg_weekly_cost: response.data.metrics.avg_weekly_cost,
        service_level: response.data.metrics.service_level,
        service_level_change: response.data.metrics.service_level_change
      },
      time_series: response.data.time_series.map(point => ({
        week: point.week,
        inventory: point.inventory,
        order: point.order,
        cost: point.cost,
        backlog: point.backlog,
        demand: point.demand,
        supply: point.supply
      }))
    };
    
    return formattedData;
  } catch (error) {
    console.error('Error fetching human dashboard:', error);
    
    // Return mock data in case of error for development
    if (process.env.NODE_ENV === 'development') {
      console.warn('Using mock dashboard data due to error');
      return {
        game_name: 'Demo Game',
        current_round: 5,
        player_role: 'RETAILER',
        metrics: {
          current_inventory: 42,
          inventory_change: 5.5,
          backlog: 12,
          total_cost: 1250.75,
          avg_weekly_cost: 250.15,
          service_level: 0.85,
          service_level_change: 0.02
        },
        time_series: Array.from({ length: 5 }, (_, i) => ({
          week: i + 1,
          inventory: Math.floor(Math.random() * 50) + 20,
          order: Math.floor(Math.random() * 30) + 10,
          cost: Math.floor(Math.random() * 300) + 100,
          backlog: Math.floor(Math.random() * 20) + 5,
          demand: Math.floor(Math.random() * 40) + 5,
          supply: Math.floor(Math.random() * 40) + 5
        })),
        last_updated: new Date().toISOString()
      };
    }
    
    throw error;
  }
};

/**
 * Format time series data for chart display
 * @param {Array} timeSeries - Array of TimeSeriesPoint objects from API
 * @param {string} role - Player's role in the game
 * @returns {Array} Formatted data for chart
 */
export const formatChartData = (timeSeries, role) => {
  if (!timeSeries || !timeSeries.length) return [];
  
  return timeSeries.map(point => ({
    week: point.week,
    inventory: point.inventory,
    order: point.order,
    cost: point.cost,
    backlog: point.backlog,
    demand: role === 'RETAILER' || role === 'MANUFACTURER' || role === 'DISTRIBUTOR' ? point.demand : undefined,
    supply: role === 'SUPPLIER' || role === 'MANUFACTURER' || role === 'DISTRIBUTOR' ? point.supply : undefined
  }));
};

