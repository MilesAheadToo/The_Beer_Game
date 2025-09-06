#!/bin/bash

# Reset the admin password directly in the database
docker-compose exec db mysql -u beer_user -p'Daybreak@2025' -e "
  USE beer_game;
  UPDATE users 
  SET hashed_password = '\$2b\$12\$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW'
  WHERE username = 'admin';
  
  SELECT 'Password reset complete' AS message;
"
