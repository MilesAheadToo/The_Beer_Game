class WebSocketService {
  constructor() {
    this.socket = null;
    this.callbacks = [];
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second delay
  }

  connect(gameId) {
    if (this.socket) {
      this.disconnect();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const token = localStorage.getItem('access_token');
    
    this.socket = new WebSocket(`${protocol}//${host}/ws/games/${gameId}?token=${token}`);

    this.socket.onopen = () => {
      console.log('WebSocket Connected');
      this.connected = true;
      this.reconnectAttempts = 0;
      this.reconnectDelay = 1000;
      this.notifyCallbacks('connected', { connected: true });
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.notifyCallbacks('message', data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.socket.onclose = (event) => {
      console.log('WebSocket Disconnected', event);
      this.connected = false;
      this.notifyCallbacks('disconnected', { connected: false });
      this.attemptReconnect(gameId);
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket Error:', error);
      this.notifyCallbacks('error', { error });
    };
  }

  attemptReconnect(gameId) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
      
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        if (!this.connected) {
          this.connect(gameId);
        }
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
      this.notifyCallbacks('reconnect_failed', { message: 'Max reconnection attempts reached' });
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.connected = false;
    }
  }

  send(message) {
    if (this.socket && this.connected) {
      this.socket.send(JSON.stringify(message));
      return true;
    }
    console.error('WebSocket is not connected');
    return false;
  }

  subscribe(callback) {
    this.callbacks.push(callback);
    // Return unsubscribe function
    return () => {
      this.callbacks = this.callbacks.filter(cb => cb !== callback);
    };
  }

  notifyCallbacks(event, data) {
    this.callbacks.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('Error in WebSocket callback:', error);
      }
    });
  }

  isConnected() {
    return this.connected;
  }
}

// Create a singleton instance
export const webSocketService = new WebSocketService();
