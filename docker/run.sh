#!/bin/bash 

ARCH=${1:-"amd64"}  # arm64 or amd64
OS=$2               # operating system name and version 

docker buildx build --platform linux/arm64  --build-arg PYTHON_VERSION=3.12 --build-arg OS_VERSION=24.04 -t pyember-ubuntu.24.04-arm64 -f docker/ubuntu.Dockerfile --load --no-cache .  git:main*

