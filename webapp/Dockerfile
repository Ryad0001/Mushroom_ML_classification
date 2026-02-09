FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
# Commande spécifique pour lancer Streamlit sur le port 8081
CMD streamlit run api.py --server.port 8081 --server.address 0.0.0.0
