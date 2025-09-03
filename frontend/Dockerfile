# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Install system dependencies for native modules
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./

# Install all dependencies including peer dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the application
COPY . .

# Set environment to production
ENV NODE_ENV=production

# Build the application
RUN npm run build

# Remove development dependencies after build
RUN npm prune --production

# Production stage
FROM nginx:alpine

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy built assets from the build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create a non-root user and set permissions
RUN adduser -D -g 'www' www \
    && chown -R www:www /usr/share/nginx/html \
    && chown -R www:www /var/cache/nginx \
    && chown -R www:www /var/log/nginx \
    && chown -R www:www /etc/nginx/conf.d

# Switch to non-root user
USER www

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
