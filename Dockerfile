# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python packages, ignoring hash checks
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-deps -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
