import React from 'react';
import { 
  Box, 
  FormControl, 
  FormLabel, 
  Grid,
  NumberInput, 
  NumberInputField, 
  NumberInputStepper, 
  NumberIncrementStepper, 
  NumberDecrementStepper,
  VStack,
  Text,
  Card,
  CardHeader,
  CardBody,
  Heading,
  useColorModeValue
} from '@chakra-ui/react';

const PricingConfigForm = ({ pricingConfig, onChange }) => {
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  const handlePriceChange = (role, field, value) => {
    const newPricingConfig = {
      ...pricingConfig,
      [role]: {
        ...pricingConfig[role],
        [field]: parseFloat(value) || 0
      }
    };
    onChange(newPricingConfig);
  };

  const renderRolePricing = (role, label) => (
    <Card variant="outline" borderColor={borderColor} mb={4}>
      <CardHeader pb={2}>
        <Heading size="sm">{label} Pricing</Heading>
      </CardHeader>
      <CardBody pt={0}>
        <Grid templateColumns={{ base: '1fr', md: '1fr 1fr' }} gap={4}>
          <FormControl>
            <FormLabel>Selling Price</FormLabel>
            <NumberInput 
              min={0.01} 
              step={0.01}
              precision={2}
              value={pricingConfig[role]?.selling_price || ''}
              onChange={(value) => handlePriceChange(role, 'selling_price', value)}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
          </FormControl>
          
          <FormControl>
            <FormLabel>Standard Cost</FormLabel>
            <NumberInput 
              min={0.01} 
              step={0.01}
              precision={2}
              value={pricingConfig[role]?.standard_cost || ''}
              onChange={(value) => handlePriceChange(role, 'standard_cost', value)}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
          </FormControl>
        </Grid>
        
        {pricingConfig[role]?.selling_price > 0 && pricingConfig[role]?.standard_cost > 0 && (
          <Box mt={2}>
            <Text fontSize="sm" color="gray.500">
              Margin: ${(pricingConfig[role].selling_price - pricingConfig[role].standard_cost).toFixed(2)} 
              ({(pricingConfig[role].selling_price > 0 ? 
                  ((pricingConfig[role].selling_price - pricingConfig[role].standard_cost) / pricingConfig[role].selling_price * 100).toFixed(1) : 
                  '0.0')}%)
            </Text>
          </Box>
        )}
      </CardBody>
    </Card>
  );

  return (
    <Card variant="outline" bg={cardBg} borderColor={borderColor} mb={6}>
      <CardHeader>
        <Heading size="md">Pricing Configuration</Heading>
        <Text color="gray.500" fontSize="sm">
          Configure pricing for each role in the supply chain
        </Text>
      </CardHeader>
      <CardBody pt={0}>
        <VStack spacing={4} align="stretch">
          {renderRolePricing('retailer', 'Retailer')}
          {renderRolePricing('wholesaler', 'Wholesaler')}
          {renderRolePricing('distributor', 'Distributor')}
          {renderRolePricing('factory', 'Factory')}
        </VStack>
      </CardBody>
    </Card>
  );
};

export default PricingConfigForm;
