#!/bin/sh

# Replace environment variables in nginx config
envsubst '${DATA_SERVICE_PORT} ${NHITS_SERVICE_PORT} ${RAG_SERVICE_PORT} ${LLM_SERVICE_PORT} ${TCN_SERVICE_PORT} ${HMM_SERVICE_PORT} ${EMBEDDER_SERVICE_PORT} ${WATCHDOG_SERVICE_PORT} ${CANDLESTICK_SERVICE_PORT} ${CANDLESTICK_TRAIN_SERVICE_PORT}' \
    < /etc/nginx/nginx.conf.template \
    > /etc/nginx/nginx.conf

# Replace port values in HTML files
sed -i "s/Port<\/span><span>3001/Port<\/span><span>${DATA_SERVICE_PORT}/g" /usr/share/nginx/html/index.html
sed -i "s/Port<\/span><span>3002/Port<\/span><span>${NHITS_SERVICE_PORT}/g" /usr/share/nginx/html/index.html
sed -i "s/Port<\/span><span>3003/Port<\/span><span>${RAG_SERVICE_PORT}/g" /usr/share/nginx/html/index.html
sed -i "s/Port<\/span><span>3004/Port<\/span><span>${LLM_SERVICE_PORT}/g" /usr/share/nginx/html/index.html

# Replace port values in JavaScript
sed -i "s/'data', ${DATA_SERVICE_PORT}/'data', ${DATA_SERVICE_PORT}/g" /usr/share/nginx/html/index.html
sed -i "s/'nhits', ${NHITS_SERVICE_PORT}/'nhits', ${NHITS_SERVICE_PORT}/g" /usr/share/nginx/html/index.html
sed -i "s/'rag', ${RAG_SERVICE_PORT}/'rag', ${RAG_SERVICE_PORT}/g" /usr/share/nginx/html/index.html
sed -i "s/'llm', ${LLM_SERVICE_PORT}/'llm', ${LLM_SERVICE_PORT}/g" /usr/share/nginx/html/index.html

# Start nginx
exec nginx -g 'daemon off;'
