FROM python:3.12-slim

WORKDIR /app

# Install dependencies

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code

COPY . .

# Command to run the model training script

CMD ["python", "python/train.py"]
