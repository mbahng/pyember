#!/bin/bash 

# ubuntu arm
docker buildx build --platform linux/arm64  --build-arg PYTHON_VERSION=3.12 --build-arg OS_VERSION=24.04 -t pyember-ubuntu.24.04-arm64 -f docker/ubuntu_arm.Dockerfile --load --no-cache .
docker buildx build --platform linux/arm64  --build-arg PYTHON_VERSION=3.12 --build-arg OS_VERSION=22.04 -t pyember-ubuntu.22.04-arm64 -f docker/ubuntu_arm.Dockerfile --load --no-cache .
docker buildx build --platform linux/arm64  --build-arg PYTHON_VERSION=3.12 --build-arg OS_VERSION=20.04 -t pyember-ubuntu.20.04-arm64 -f docker/ubuntu_arm.Dockerfile --load --no-cache .

# ubuntu x86 
docker buildx build --platform linux/amd64  --build-arg PYTHON_VERSION=3.12 --build-arg OS_VERSION=24.04 -t pyember-ubuntu.24.04-amd64 -f docker/ubuntu_amd.Dockerfile --load --no-cache .
docker buildx build --platform linux/amd64  --build-arg PYTHON_VERSION=3.12 --build-arg OS_VERSION=22.04 -t pyember-ubuntu.22.04-amd64 -f docker/ubuntu_amd.Dockerfile --load --no-cache .
docker buildx build --platform linux/amd64  --build-arg PYTHON_VERSION=3.12 --build-arg OS_VERSION=20.04 -t pyember-ubuntu.20.04-amd64 -f docker/ubuntu_amd.Dockerfile --load --no-cache .

