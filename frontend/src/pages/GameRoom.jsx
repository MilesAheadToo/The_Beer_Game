import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  ArrowPathIcon, 
  ArrowUturnLeftIcon,
  ChatBubbleLeftRightIcon,
  Cog6ToothIcon,
  UserGroupIcon,
  ChartBarIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';
import gameApi from '../services/gameApi';
import { mixedGameApi } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import { toast } from 'react-toastify';

const GameRoom = () => {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [game, setGame] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTab, setActiveTab] = useState('game');
  const [orderAmount, setOrderAmount] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [message, setMessage] = useState('');
  const chatEndRef = useRef(null);
  const ws = useRef(null);

  // Fetch game data
  const fetchGame = async () => {
    try {
      const gameData = await gameApi.getGame(gameId);
      setGame(gameData);
      return gameData;
    } catch (error) {
      console.error('Failed to fetch game:', error);
      toast.error('Failed to load game. It may not exist or you may not have permission.');
      navigate('/');
    } finally {
      setIsLoading(false);
    }
  };

  // Set up WebSocket connection
  useEffect(() => {
    // Initialize WebSocket connection
    const setupWebSocket = () => {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ws/game/${gameId}/`;
      
      ws.current = new WebSocket(wsUrl);
      
      ws.current.onopen = async () => {
        console.log('WebSocket connected');
        // Obtain a fresh access token via cookie-based refresh and authenticate WS
        try {
          const { access_token } = await mixedGameApi.refreshToken();
          if (access_token) {
            ws.current.send(JSON.stringify({ type: 'authenticate', token: access_token }));
          }
        } catch (e) {
          console.warn('WS auth token refresh failed:', e?.message || e);
        }
      };
      
      ws.current.onmessage = (e) => {
        const data = JSON.parse(e.data);
        console.log('WebSocket message received:', data);
        
        switch (data.type) {
          case 'game_update':
            setGame(data.game);
            break;
            
          case 'chat_message':
            setChatMessages(prev => [...prev, data.message]);
            break;
            
          case 'error':
            toast.error(data.message);
            break;
            
          default:
            console.warn('Unknown message type:', data.type);
        }
      };
      
      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        // Try to reconnect after a delay
        setTimeout(() => {
          if (ws.current) setupWebSocket();
        }, 3000);
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };
    
    setupWebSocket();
    
    // Initial data fetch
    fetchGame();
    
    // Clean up WebSocket on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    };
  }, [gameId, navigate]);
  
  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages]);

  const handleSubmitOrder = async (e) => {
    e.preventDefault();
    if (!orderAmount || isNaN(orderAmount) || orderAmount < 0) {
      toast.error('Please enter a valid order amount');
      return;
    }
    
    try {
      setIsSubmitting(true);
      await gameApi.submitOrder(gameId, { amount: parseInt(orderAmount, 10) });
      setOrderAmount('');
      toast.success('Order submitted successfully');
    } catch (error) {
      console.error('Failed to submit order:', error);
      toast.error(error.response?.data?.detail || 'Failed to submit order');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!message.trim()) return;
    
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: 'chat_message',
        message: message.trim(),
        sender: user.username
      }));
      setMessage('');
    }
  };
  
  const startGame = async () => {
    try {
      setIsSubmitting(true);
      await gameApi.startGame(gameId);
      toast.success('Game started!');
    } catch (error) {
      console.error('Failed to start game:', error);
      toast.error(error.response?.data?.detail || 'Failed to start game');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const leaveGame = async () => {
    try {
      await gameApi.leaveGame(gameId);
      navigate('/');
      toast.success('Left the game');
    } catch (error) {
      console.error('Failed to leave game:', error);
      toast.error('Failed to leave game');
    }
  };
  
  if (isLoading || !game) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
      </div>
    );
  }
  
  const currentPlayer = game.players.find(p => p.user_id === user.id);
  const isGameMaster = game.created_by === user.id;
  const isGameActive = game.status === 'in_progress';
  const isPlayerReady = currentPlayer?.is_ready;
  const allPlayersReady = game.players.every(p => p.is_ready) && game.players.length >= 2;
  
  // Render game board based on player role
  const renderGameBoard = () => {
    if (!isGameActive) {
      return (
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <h2 className="text-xl font-semibold mb-4">Waiting for game to start</h2>
          <p className="text-gray-600 mb-6">
            {isGameMaster 
              ? 'You are the game master. Start the game when all players are ready.'
              : 'The game will start once the game master begins.'}
          </p>
          
          <div className="space-y-4 max-w-md mx-auto">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium mb-2">Players ({game.players.length}/{game.max_players})</h3>
              <ul className="space-y-2">
                {game.players.map(player => (
                  <li key={player.id} className="flex items-center justify-between">
                    <span className={`${player.is_ready ? 'text-green-600' : 'text-gray-600'}`}>
                      {player.username}
                      {player.is_ready && ' ✓'}
                      {player.user_id === game.created_by && ' 👑'}
                    </span>
                    {player.user_id === user.id && !isPlayerReady && (
                      <button
                        onClick={async () => {
                          try {
                            await gameApi.setPlayerReady(gameId, { is_ready: true });
                            toast.success('You are ready!');
                          } catch (error) {
                            toast.error('Failed to update status');
                          }
                        }}
                        className="text-sm bg-green-500 text-white px-3 py-1 rounded hover:bg-green-600"
                      >
                        I'm Ready
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </div>
            
            {isGameMaster && (
              <div className="pt-4">
                <button
                  onClick={startGame}
                  disabled={!allPlayersReady || isSubmitting}
                  className={`w-full py-2 px-4 rounded-md text-white font-medium ${
                    allPlayersReady 
                      ? 'bg-indigo-600 hover:bg-indigo-700' 
                      : 'bg-gray-400 cursor-not-allowed'
                  }`}
                >
                  {isSubmitting ? 'Starting...' : 'Start Game'}
                </button>
              </div>
            )}
          </div>
        </div>
      );
    }
    
    // Render actual game board when game is in progress
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold">{game.name}</h2>
          <div className="flex items-center space-x-4">
            <span className="px-3 py-1 bg-blue-100 text-blue-800 text-sm font-medium rounded-full">
              Round {game.current_round} of {game.settings.max_rounds}
            </span>
            <span className="px-3 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
              ${game.current_balance || 0}
            </span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Player's inventory and orders */}
          <div className="md:col-span-2 space-y-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-medium mb-3">Your Supply Chain</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold">{currentPlayer?.inventory || 0}</div>
                  <div className="text-sm text-gray-500">Inventory</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{currentPlayer?.backlog || 0}</div>
                  <div className="text-sm text-gray-500">Backlog</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{currentPlayer?.incoming_order || 0}</div>
                  <div className="text-sm text-gray-500">Incoming</div>
                </div>
              </div>
              
              <form onSubmit={handleSubmitOrder} className="mt-4">
                <label htmlFor="orderAmount" className="block text-sm font-medium text-gray-700 mb-1">
                  Place Order (0-{currentPlayer?.inventory + (currentPlayer?.incoming_order || 0) + 10})
                </label>
                <div className="flex space-x-2">
                  <input
                    type="number"
                    id="orderAmount"
                    min="0"
                    max={currentPlayer?.inventory + (currentPlayer?.incoming_order || 0) + 10}
                    value={orderAmount}
                    onChange={(e) => setOrderAmount(e.target.value)}
                    className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                    placeholder="Enter amount"
                  />
                  <button
                    type="submit"
                    disabled={isSubmitting || !isGameActive}
                    className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                  >
                    {isSubmitting ? 'Submitting...' : 'Order'}
                  </button>
                </div>
              </form>
            </div>
            
            {/* Game status and history */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="font-medium mb-3">Game Status</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Current Round:</span>
                  <span className="font-medium">{game.current_round} / {game.settings.max_rounds}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Time Left:</span>
                  <span className="font-medium">2:30</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Your Score:</span>
                  <span className="font-medium">{currentPlayer?.score || 0} pts</span>
                </div>
              </div>
              
              <h4 className="font-medium mt-4 mb-2">Recent Orders</h4>
              <div className="bg-gray-50 p-3 rounded max-h-32 overflow-y-auto">
                {game.recent_orders?.length > 0 ? (
                  <ul className="space-y-1">
                    {game.recent_orders.map((order, idx) => (
                      <li key={idx} className="text-sm">
                        <span className="font-medium">{order.player}:</span> Ordered {order.amount} units
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500">No orders placed yet</p>
                )}
              </div>
            </div>
          </div>
          
          {/* Players list */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-medium mb-3">Players</h3>
            <ul className="space-y-3">
              {game.players.map((player) => (
                <li 
                  key={player.id} 
                  className={`p-3 rounded ${player.user_id === user.id ? 'bg-indigo-50 border border-indigo-100' : 'bg-white'}`}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <span className="font-medium">
                        {player.username}
                        {player.user_id === game.created_by && ' 👑'}
                      </span>
                      <p className="text-sm text-gray-500">
                        Score: {player.score || 0}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-sm">
                        Inv: {player.inventory || 0}
                      </div>
                      <div className="text-sm text-gray-500">
                        Bklog: {player.backlog || 0}
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto pad-8">
      <div className="flex items-center justify-between mb-6">
        <button
          onClick={() => navigate('/')}
          className="inline-flex items-center text-sm text-gray-600 hover:text-gray-900"
        >
          <ArrowUturnLeftIcon className="h-5 w-5 mr-1" />
          Back to Lobby
        </button>
        
        <div className="flex space-x-2">
          <button
            onClick={() => window.location.reload()}
            className="p-2 text-gray-500 hover:text-gray-700"
            title="Refresh"
          >
            <ArrowPathIcon className="h-5 w-5" />
          </button>
          <button
            onClick={leaveGame}
            className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
          >
            Leave Game
          </button>
        </div>
      </div>
      
      <div className="flex flex-col md:flex-row gap-6">
        {/* Main game area */}
        <div className="flex-1">
          {renderGameBoard()}
        </div>
        
        {/* Right sidebar */}
        <div className="w-full md:w-80 flex-shrink-0">
          {/* Tabs */}
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-4">
              <button
                onClick={() => setActiveTab('chat')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'chat'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <ChatBubbleLeftRightIcon className="h-5 w-5 inline-block mr-1" />
                Chat
              </button>
              <button
                onClick={() => setActiveTab('players')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'players'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <UserGroupIcon className="h-5 w-5 inline-block mr-1" />
                Players
              </button>
              <button
                onClick={() => setActiveTab('stats')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'stats'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <ChartBarIcon className="h-5 w-5 inline-block mr-1" />
                Stats
              </button>
            </nav>
          </div>
          
          {/* Tab content */}
          <div className="mt-2">
            {activeTab === 'chat' && (
              <div className="bg-white rounded-lg border border-gray-200 overflow-hidden flex flex-col h-96">
                <div className="p-3 border-b border-gray-200">
                  <h3 className="text-sm font-medium">Game Chat</h3>
                </div>
                <div className="flex-1 overflow-y-auto p-3 space-y-3">
                  {chatMessages.length > 0 ? (
                    chatMessages.map((msg, idx) => (
                      <div key={idx} className="text-sm">
                        <span className="font-medium">{msg.sender}:</span>{' '}
                        <span>{msg.message}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-center text-gray-500 text-sm h-full flex items-center justify-center">
                      No messages yet. Say hello!
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
                <div className="p-3 border-t border-gray-200">
                  <form onSubmit={handleSendMessage} className="flex">
                    <input
                      type="text"
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      placeholder="Type a message..."
                      className="flex-1 rounded-l-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                    />
                    <button
                      type="submit"
                      className="inline-flex items-center px-3 py-2 border border-l-0 border-gray-300 bg-indigo-600 text-white rounded-r-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      Send
                    </button>
                  </form>
                </div>
              </div>
            )}
            
            {activeTab === 'players' && (
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <h3 className="font-medium mb-3">Players ({game.players.length}/{game.max_players})</h3>
                <ul className="space-y-2">
                  {game.players.map((player) => (
                    <li key={player.id} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded">
                      <div className="flex items-center">
                        <div className={`h-2 w-2 rounded-full mr-2 ${
                          player.is_online ? 'bg-green-500' : 'bg-gray-300'
                        }`}></div>
                        <span className={`${player.user_id === user.id ? 'font-medium text-indigo-600' : ''}`}>
                          {player.username}
                          {player.user_id === game.created_by && ' 👑'}
                        </span>
                      </div>
                      <span className="text-sm text-gray-500">
                        {player.score || 0} pts
                      </span>
                    </li>
                  ))}
                </ul>
                
                {isGameMaster && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h4 className="text-sm font-medium mb-2">Game Master Controls</h4>
                    <div className="space-y-2">
                      <button
                        onClick={startGame}
                        disabled={!allPlayersReady || isSubmitting || isGameActive}
                        className="w-full text-left px-3 py-1.5 text-sm rounded bg-indigo-100 text-indigo-700 hover:bg-indigo-200 disabled:opacity-50"
                      >
                        Start Game
                      </button>
                      <button
                        onClick={async () => {
                          if (window.confirm('Are you sure you want to end the game?')) {
                            try {
                              await gameApi.endGame(gameId);
                              toast.success('Game ended');
                            } catch (error) {
                              toast.error('Failed to end game');
                            }
                          }
                        }}
                        disabled={!isGameActive}
                        className="w-full text-left px-3 py-1.5 text-sm rounded bg-red-100 text-red-700 hover:bg-red-200 disabled:opacity-50"
                      >
                        End Game
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {activeTab === 'stats' && (
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <h3 className="font-medium mb-3">Game Statistics</h3>
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-1">Current Round</h4>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className="bg-blue-600 h-2.5 rounded-full" 
                        style={{ width: `${(game.current_round / game.settings.max_rounds) * 100}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>0</span>
                      <span>Round {game.current_round} of {game.settings.max_rounds}</span>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Leaderboard</h4>
                    <ol className="space-y-2">
                      {[...game.players]
                        .sort((a, b) => (b.score || 0) - (a.score || 0))
                        .slice(0, 3)
                        .map((player, idx) => (
                          <li key={player.id} className="flex items-center">
                            <span className="text-gray-500 w-6">{idx + 1}.</span>
                            <span className="flex-1">{player.username}</span>
                            <span className="font-medium">{player.score || 0} pts</span>
                          </li>
                        ))}
                    </ol>
                  </div>
                  
                  <div className="pt-2 border-t border-gray-200">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Game Info</h4>
                    <dl className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <dt className="text-gray-500">Round Time:</dt>
                        <dd>{game.settings.round_duration} sec</dd>
                      </div>
                      <div className="flex justify-between text-sm">
                        <dt className="text-gray-500">Max Rounds:</dt>
                        <dd>{game.settings.max_rounds}</dd>
                      </div>
                      <div className="flex justify-between text-sm">
                        <dt className="text-gray-500">Starting Inventory:</dt>
                        <dd>{game.settings.starting_inventory}</dd>
                      </div>
                    </dl>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameRoom;
