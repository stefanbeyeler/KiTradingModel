#!/bin/bash
# =============================================================================
# Deploy Services with Git Version Info
# =============================================================================
# This script exports Git version information and starts the microservices.
# Run from the project root directory.
#
# Usage:
#   ./scripts/deploy-with-version.sh              # Start all services
#   ./scripts/deploy-with-version.sh --build      # Rebuild and start all services
#   ./scripts/deploy-with-version.sh data-service # Start specific service
# =============================================================================

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Export Git version info
export BUILD_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')
export BUILD_DATE=$(git log -1 --format=%cI 2>/dev/null || date -Iseconds)
export BUILD_NUMBER=$(git rev-list --count HEAD 2>/dev/null || echo '0')

# Build version string: 1.0.0.<commit_count>+<short_hash>
TAG=$(git describe --tags --exact-match 2>/dev/null || echo '')
if [ -n "$TAG" ]; then
    export BUILD_VERSION="${TAG#v}"
else
    export BUILD_VERSION="1.0.0.${BUILD_NUMBER}+${BUILD_COMMIT}"
fi

echo "==================================================="
echo "Deploying with version: $BUILD_VERSION"
echo "Commit: $BUILD_COMMIT"
echo "Build number: $BUILD_NUMBER"
echo "Build date: $BUILD_DATE"
echo "==================================================="

# Handle arguments
if [ "$1" == "--build" ]; then
    echo "Rebuilding and starting all services..."
    docker-compose -f docker-compose.microservices.yml up -d --build
elif [ -n "$1" ]; then
    echo "Starting service: $1..."
    docker-compose -f docker-compose.microservices.yml up -d "$1"
else
    echo "Starting all services..."
    docker-compose -f docker-compose.microservices.yml up -d
fi

echo ""
echo "Done! Services are starting with version $BUILD_VERSION"
echo "Check status with: docker ps"
