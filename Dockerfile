FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY price_tag_pipeline ./price_tag_pipeline
COPY tools ./tools
RUN mkdir -p models

RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["price-tag-pipeline"]
CMD ["--help"]
