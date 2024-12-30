# Use Python 3.12
FROM python:3.12-slim

# Install system dependencies including CMake, C++ compiler, and Google Test
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    libgtest-dev \
    && cd /usr/src/gtest \
    && cmake CMakeLists.txt \
    && make \
    && cp lib/*.a /usr/lib \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /pyember

# Set environment variables for compilation flags
ENV CXXFLAGS="-fPIC"
ENV CFLAGS="-fPIC"

# Copy the entire project structure 
COPY . .

# Install Python dependencies and build the package
RUN pip install .

# Run tests (using your test script)
CMD ["./run_tests.sh", "all"]
