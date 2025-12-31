#!/bin/bash

echo "Cleaning macOS metadata files..."
find . -name "._*" -type f -delete

echo "Building Docker image for linux/amd64..."
docker build --platform linux/amd64 -f scripts/setup/Dockerfile -t biremurhan/image-text-contrast:v0.4 .