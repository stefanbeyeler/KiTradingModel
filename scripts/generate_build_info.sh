#!/bin/bash
# Generate build info file for Docker builds
# This script creates a build_info.json file with Git information
# that can be read by the application at runtime

set -e

OUTPUT_FILE="${1:-src/build_info.json}"

# Get Git information (with fallbacks)
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "")
GIT_COMMIT_COUNT=$(git rev-list --count HEAD 2>/dev/null || echo "0")
GIT_TAG=$(git describe --tags --exact-match HEAD 2>/dev/null || echo "")
GIT_DATE=$(git log -1 --format=%cI 2>/dev/null || date -Iseconds)
BUILD_DATE=$(date -Iseconds)

# Build version string
if [ -n "$GIT_TAG" ]; then
    VERSION="${GIT_TAG#v}"
elif [ -n "$GIT_COMMIT" ]; then
    VERSION="1.0.0.${GIT_COMMIT_COUNT}+${GIT_COMMIT}"
else
    VERSION="1.0.0"
fi

# Create JSON file
cat > "$OUTPUT_FILE" << EOF
{
    "version": "$VERSION",
    "commit": "$GIT_COMMIT",
    "commit_count": "$GIT_COMMIT_COUNT",
    "tag": "$GIT_TAG",
    "commit_date": "$GIT_DATE",
    "build_date": "$BUILD_DATE"
}
EOF

echo "Generated $OUTPUT_FILE with version: $VERSION"
