# Use a slim Python image for a smaller footprint
FROM python:3.12-slim

# Install uv to manage dependencies efficiently
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies exactly as defined in the lockfile
# --frozen ensures uv.lock is not updated during build
RUN uv sync --frozen --no-cache --no-install-project

# Copy the source code and the dataset needed at runtime
# Note: only bank-full.csv is included; all other data files are excluded via .dockerignore
COPY src/ ./src/
COPY data/bank-full.csv ./data/bank-full.csv

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit application
# The --server.address=0.0.0.0 is mandatory for Docker to route traffic
ENTRYPOINT ["uv", "run", "streamlit", "run", "src/mi_paquete/app/dashboard.py", "--server.address=0.0.0.0"]
