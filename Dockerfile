FROM python:3.9

# Create user to run the app
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Expose port 7860 (HuggingFace requirement)
EXPOSE 7860

# Run the Flask application
CMD ["python", "app.py"]
