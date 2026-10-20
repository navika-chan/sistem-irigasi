# Lightweight Dockerfile untuk Layanan Cloud Rekomendasi Irigasi Presisi
FROM python:3.10-slim

# Variabel Lingkungan
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Direktori Kerja
WORKDIR /app

# Salin & Pasang Dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin Kode Aplikasi dan Berkas Bobot JSON
COPY main.py .
COPY models/best_rnn_weights.json ./models/

# Port Default
EXPOSE 8000

# Perintah Eksekusi Server (Kompatibel dengan Dynamic PORT Render)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
