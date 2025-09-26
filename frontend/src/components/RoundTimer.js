import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Text,
  Progress,
  VStack,
  HStack,
  Button,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  useToast,
  Badge,
  Textarea,
} from '@chakra-ui/react';
import { CheckCircleIcon, TimeIcon, WarningIcon } from '@chakra-ui/icons';
import mixedGameApi from '../services/api';

const RoundTimer = ({
  gameId,
  playerId,
  roundNumber,
  onOrderSubmit,
  isPlayerTurn,
  orderComment = '',
  onCommentChange,
  readOnly = false,
}) => {
  const [timeLeft, setTimeLeft] = useState(60);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [orderQuantity, setOrderQuantity] = useState(0);
  const [roundEndsAt, setRoundEndsAt] = useState(null);
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const toast = useToast();
  const timerRef = useRef(null);

  const instructionText = readOnly
    ? 'Viewing order entry details for this role'
    : 'Place your order for the next round';

  // Handle order submission
  const handleSubmit = useCallback(async (quantity) => {
    if (readOnly) {
      return;
    }

    if (quantity === null || quantity < 0) return;

    setIsSubmitting(true);
    try {
      await onOrderSubmit(quantity, orderComment);
      setHasSubmitted(true);
      toast({
        title: 'Order submitted!',
        status: 'success',
        duration: 2000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error submitting order:', error);
      toast({
        title: 'Error submitting order',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [readOnly, onOrderSubmit, orderComment, toast]);

  // Fetch round status when component mounts or round changes
  useEffect(() => {
    const fetchRoundStatus = async () => {
      try {
        const status = await mixedGameApi.getRoundStatus(gameId);
        setRoundEndsAt(new Date(status.ends_at));
        
        // Check if player has already submitted
        if (status.submitted_players?.some(p => p.id === playerId)) {
          setHasSubmitted(true);
          const playerOrder = status.submitted_players.find(p => p.id === playerId);
          if (playerOrder) {
            setOrderQuantity(playerOrder.quantity);
            if (playerOrder.comment && onCommentChange) {
              onCommentChange(playerOrder.comment);
            }
          }
        }
      } catch (error) {
        console.error('Error fetching round status:', error);
      }
    };

    fetchRoundStatus();
    
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [gameId, playerId, roundNumber, onCommentChange]);

  // Set up timer
  useEffect(() => {
    if (!roundEndsAt) return;
    
    const updateTimer = () => {
      const now = new Date();
      const diff = Math.max(0, Math.floor((roundEndsAt - now) / 1000));
      setTimeLeft(diff);
      
      // If time's up and we haven't submitted, submit zero
      if (diff <= 0 && !hasSubmitted && isPlayerTurn) {
        handleSubmit(0);
      }
    };
    
    // Initial update
    updateTimer();
    
    // Set up interval
    timerRef.current = setInterval(updateTimer, 1000);
    
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [roundEndsAt, hasSubmitted, isPlayerTurn, handleSubmit]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  const progressValue = (timeLeft / 60) * 100; // Assuming 60 seconds per round
  
  return (
    <VStack spacing={4} p={4} borderWidth="1px" borderRadius="lg" bg="white" boxShadow="sm">
      <HStack width="100%" justifyContent="space-between">
        <Text fontSize="lg" fontWeight="bold">
          Round {roundNumber}
        </Text>
        <HStack>
          <TimeIcon color={timeLeft < 10 ? 'red.500' : 'gray.500'} />
          <Text fontWeight="semibold" color={timeLeft < 10 ? 'red.500' : 'gray.700'}>
            {formatTime(timeLeft)}
          </Text>
          {hasSubmitted ? (
            <Badge colorScheme="green" p={1} borderRadius="md">
              <HStack spacing={1}>
                <CheckCircleIcon />
                <Text>Submitted: {orderQuantity}</Text>
              </HStack>
            </Badge>
          ) : (
            <Badge colorScheme={isPlayerTurn ? 'yellow' : 'gray'} p={1} borderRadius="md">
              {readOnly
                ? isPlayerTurn
                  ? 'Active'
                  : 'Waiting'
                : isPlayerTurn
                  ? 'Your Turn'
                  : 'Waiting...'}
            </Badge>
          )}
        </HStack>
      </HStack>
      
      <Progress 
        value={progressValue} 
        size="sm" 
        width="100%" 
        colorScheme={timeLeft < 10 ? 'red' : 'green'}
        borderRadius="full"
      />
      
      {((isPlayerTurn && !hasSubmitted) || readOnly) && (
        <VStack width="100%" spacing={4} mt={4}>
          <Text fontSize="sm" color="gray.600">
            {instructionText}
          </Text>
          <HStack width="100%" alignItems="flex-start" spacing={3}>
            <NumberInput
              min={0}
              value={orderQuantity}
              onChange={(value) => setOrderQuantity(parseInt(value) || 0)}
              width="140px"
              isDisabled={readOnly}
            >
              <NumberInputField />
              <NumberInputStepper>
                <NumberIncrementStepper />
                <NumberDecrementStepper />
              </NumberInputStepper>
            </NumberInput>
            <Textarea
              value={orderComment}
              onChange={(event) => onCommentChange?.(event.target.value)}
              placeholder="Why are you ordering this amount?"
              resize="vertical"
              flex="1"
              minH="80px"
              isDisabled={readOnly}
            />
            <Button
              colorScheme="blue"
              onClick={() => handleSubmit(orderQuantity)}
              isLoading={isSubmitting}
              loadingText="Submitting..."
              isDisabled={readOnly}
            >
              Submit Order
            </Button>
          </HStack>
          {!readOnly && timeLeft < 10 && (
            <HStack color="red.500" fontSize="sm">
              <WarningIcon />
              <Text>Time is running out! Submit your order soon.</Text>
            </HStack>
          )}
        </VStack>
      )}

      {!readOnly && !isPlayerTurn && !hasSubmitted && (
        <Text fontSize="sm" color="gray.500" textAlign="center">
          Waiting for your turn to place an order...
        </Text>
      )}

      {readOnly && !isPlayerTurn && !hasSubmitted && (
        <Text fontSize="sm" color="gray.500" textAlign="center">
          Waiting for this role to place an order...
        </Text>
      )}

      {hasSubmitted && (
        <Text fontSize="sm" color={readOnly ? 'gray.600' : 'green.600'} textAlign="center">
          {readOnly
            ? `Submitted order: ${orderQuantity} units.`
            : `Your order of ${orderQuantity} units has been submitted for this round.`}
        </Text>
      )}
    </VStack>
  );
};

export default RoundTimer;
