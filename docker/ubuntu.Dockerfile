ARG ARCH=arm64
ARG OS_VERSION=22.04
ARG PYTHON_VERSION=3.12
FROM ubuntu:${OS_VERSION} 
ENV MINICONDA_ARCH=aarch64
RUN if [ "$ARCH" = "amd64" ]; then \
      ENV MINICONDA_ARCH=x86_64; \
    fi

# Install minimal system dependencies including CMake, C++ compiler
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    libgtest-dev \
    wget \ 
    ca-certificates \
    pybind11-dev \ 
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*


# Set up user and switch to it
RUN mkdir -p /user/dev 
RUN useradd --home /user/dev --shell /bin/bash dev
RUN chown -R dev:dev /user/dev
USER dev

# install conda 
RUN mkdir -p ~/miniconda3 \ 
    && echo ${MINICONDA_ARCH} \
    && wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh" -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh  

ENV PATH="/user/dev/miniconda3/bin:${PATH}"

# just install conda to base environment
RUN conda install -y python=${PYTHON_VERSION} && pip install pybind11==2.13.5

# need this for some reason so that ember detects .so file 
RUN conda install -c conda-forge libstdcxx-ng

# Set the working directory in the container
WORKDIR /user/dev/pyember

# Copy the entire project structure as root
USER root
RUN chown -R dev:dev /user/dev/pyember
COPY --chown=dev:dev . /user/dev/pyember
USER dev

# Set environment variables for compilation flags (need this?)
ENV CXXFLAGS="-fPIC"
ENV CFLAGS="-fPIC"

# Install Python dependencies and build the package 
# no debug -g (for lldb) or test modules are needed for production 
RUN CMAKE_DEBUG=0 CMAKE_DEV=0 pip install -e . -vvv 

# Run tests to ensure everything works
CMD ["./run_tests.sh", "python"]
