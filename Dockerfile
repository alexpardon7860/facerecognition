FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libglib2.0-0 \
    libgtk-3-0 \
    libsm6 \
    libice6 \
    libxrandr2 \
    libxss1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libcairo-gobject2 \
    libgdk-pixbuf2.0-0 \
    libpango-1.0-0 \
    libharfbuzz0b \
    libfontconfig1 \
    libfreetype6 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p face_database attendance_logs

# Expose port
EXPOSE 8080

# Use gunicorn to serve the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
