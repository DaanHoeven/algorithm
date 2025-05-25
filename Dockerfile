# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /tmp

RUN apt-get update && apt-get install -y ffmpeg gcc libsndfile1 libglib2.0-0 libsm6 libxext6
RUN apt-get update && apt-get install -y curl procps
# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port the Flask app will run on
EXPOSE 5001

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

RUN mkdir -p /tmp/temp_audio && chmod -R 777 /tmp/temp_audio

# Command to run the Flask application
CMD ["flask", "run"]