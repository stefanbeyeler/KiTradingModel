FROM nginx:alpine

# Copy dashboard files
COPY static/index.html /usr/share/nginx/html/index.html

# Copy nginx configuration
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
