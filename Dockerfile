# Stage 1: Builder - for installing dependencies and building artifacts
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for Python and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.9 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only dependency files first for caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install development dependencies (only if needed for dev stage or specific build steps)
# RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the application code
COPY . .

# Stage 2: Production - optimized for runtime
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as production

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code from builder stage (only necessary files for production)
COPY --from=builder /app /app

# Command to run the application (example, can be overridden)
# CMD ["python", "scripts/train.py"]

# Stage 3: Development - includes development tools and hot-reload setup
FROM production as development

# Install development-specific tools (e.g., pytest, black, ruff)
# These are already in requirements-dev.txt, so ensure they are installed in the builder stage
# if they are needed for tests/linting during development.
# For a true development container, you might install more tools here or in the builder.

# Expose ports for potential web interfaces or services during development
# EXPOSE 8000

# Default command for development (e.g., keep container running, or run a dev server)
# CMD ["bash"]

