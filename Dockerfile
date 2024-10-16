# Use the official Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the app files to the container
COPY . /app

# Install required Python packages
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]