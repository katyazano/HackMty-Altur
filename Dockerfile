# Use standard Python base image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app

# Ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Copy application script
COPY app.py .

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
