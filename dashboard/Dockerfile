FROM nginx:alpine

LABEL maintainer="KI Trading Model"
LABEL description="Web Dashboard for KI Trading Model"

# Remove default nginx config
RUN rm -rf /usr/share/nginx/html/*

# Copy dashboard files
COPY static/index.html /usr/share/nginx/html/index.html

# Copy nginx configuration
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
