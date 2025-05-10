# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# and Install Python dependencies
# We use --no-cache-dir to reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This includes app.py, Procfile, data directories, scripts etc.
COPY . .

# Make port available to the world outside this container (Streamlit default is 8501)
# Vercel will use the $PORT environment variable provided to the container.
# Streamlit needs to be told to listen on 0.0.0.0 to be accessible.
# The Procfile already handles --server.port $PORT and streamlit run is on 0.0.0.0 by default.
EXPOSE 8501

# HEALTHCHECK to ensure Streamlit app is responsive
# Vercel uses its own health checks, but this is good practice for Docker images.
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the application
# Vercel will inject the $PORT environment variable.
# Streamlit listens on 0.0.0.0 by default making it accessible.
CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.headless=true"] 