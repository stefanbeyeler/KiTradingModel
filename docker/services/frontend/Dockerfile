# Frontend Dockerfile with Nginx API Gateway
FROM nginx:alpine

LABEL maintainer="KI Trading Model"
LABEL service="frontend"

# Build arguments for service ports
ARG DATA_SERVICE_PORT=3001
ARG NHITS_SERVICE_PORT=3002
ARG RAG_SERVICE_PORT=3003
ARG LLM_SERVICE_PORT=3004
ARG TCN_SERVICE_PORT=3005
ARG HMM_SERVICE_PORT=3006
ARG EMBEDDER_SERVICE_PORT=3007

# Set environment variables
ENV DATA_SERVICE_PORT=${DATA_SERVICE_PORT}
ENV NHITS_SERVICE_PORT=${NHITS_SERVICE_PORT}
ENV RAG_SERVICE_PORT=${RAG_SERVICE_PORT}
ENV LLM_SERVICE_PORT=${LLM_SERVICE_PORT}
ENV TCN_SERVICE_PORT=${TCN_SERVICE_PORT}
ENV HMM_SERVICE_PORT=${HMM_SERVICE_PORT}
ENV EMBEDDER_SERVICE_PORT=${EMBEDDER_SERVICE_PORT}

# Copy nginx template configuration
COPY docker/services/frontend/nginx.conf.template /etc/nginx/nginx.conf.template

# Copy static HTML
COPY docker/services/frontend/html /usr/share/nginx/html

# Copy startup script
COPY docker/services/frontend/start.sh /start.sh
RUN chmod +x /start.sh

# Fix permissions
RUN chmod -R 755 /usr/share/nginx/html && \
    chmod 644 /usr/share/nginx/html/*.html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/health || exit 1

# Start script handles envsubst and nginx
CMD ["/start.sh"]
