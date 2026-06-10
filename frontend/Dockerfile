# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Ensure devDependencies are installed during build, even if NODE_ENV=production
ENV NODE_ENV=development

# Copy package files
COPY package*.json ./

# Install dependencies including devDependencies required for Vite
RUN npm ci --include=dev

# Ensure npm binaries and esbuild binary are executable inside Linux containers
RUN chmod +x ./node_modules/.bin/*
RUN chmod +x ./node_modules/vite/bin/vite.js
RUN find ./node_modules -type f -name esbuild -exec chmod +x {} +

# Copy source code
COPY . .

# Build the React app using Node directly to avoid shell permission issues
RUN node ./node_modules/vite/bin/vite.js build

# Production stage with nginx
FROM nginx:alpine

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built React app
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
