# Frontend Dockerfile with Nginx API Gateway
FROM nginx:alpine

LABEL maintainer="KI Trading Model"
LABEL service="frontend"

# Copy nginx configuration
COPY docker/services/frontend/nginx.conf /etc/nginx/nginx.conf

# Copy static HTML (placeholder - will be replaced with React/Vue later)
COPY docker/services/frontend/html /usr/share/nginx/html

# Fix permissions
RUN chmod -R 755 /usr/share/nginx/html && \
    chmod 644 /usr/share/nginx/html/*.html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/health || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
