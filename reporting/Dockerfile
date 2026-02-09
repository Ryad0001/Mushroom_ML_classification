FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir pandas numpy==1.26.4 pydantic==1.10.13 evidently && \
    apt-get purge -y --auto-remove gcc python3-dev && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.10/site-packages"
# Commande pour générer le rapport PUIS lancer l'interface
CMD python project.py && evidently ui --host 0.0.0.0 --port 8082
