FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install pandas evidently
ENV PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.10/site-packages"
# Commande pour générer le rapport PUIS lancer l'interface
CMD python project.py && evidently ui --host 0.0.0.0 --port 8082
