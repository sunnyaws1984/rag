# Sample DockerFile

FROM python:3.13

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Restore normal pip behavior
ENV PIP_NO_DEPENDENCIES=

# Copy application code
COPY retrieval_with_llm.py .

# Copy FAISS index
COPY hr_policy_faiss_index/ hr_policy_faiss_index/

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "retrieval_with_llm.py"]
