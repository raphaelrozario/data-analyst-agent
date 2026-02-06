# Using a lightweight official Python image
FROM python:3.11-slim

# Setting the working directory inside the container
WORKDIR /app

# Installation of system dependencies that are handy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copying requirements first
COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the app code
COPY . .

# Streamlit runs on 8501 by default - so expose that
EXPOSE 8501

# Make Streamlit run in headless mode
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

# Command to start your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
