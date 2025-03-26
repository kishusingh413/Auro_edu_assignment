FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 4000

# Set environment variables
ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run Flask when the container starts
CMD ["sh", "-c", "if [ ! -d migrations ]; then flask db init && flask db migrate -m 'Initial migration'; fi && flask db upgrade && gunicorn -w 4 -b 0.0.0.0:4000 server:app"]