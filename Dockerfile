FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# The model is trained once at build time, so the three services share exactly the same weights.
RUN python -m robustmon train

# One image, three roles: the command decides (produce, infer, monitor). Default: the offline demo.
CMD ["python", "-m", "robustmon", "demo", "--out", "/tmp/demo"]
