# Stage 1: build the web interface (Node.js is only needed here, never at runtime).
FROM node:22-bookworm AS ui
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html vite.config.ts tsconfig.json ./
COPY app ./app
COPY public ./public
RUN npm run build

# Stage 2: Python worker that also serves the interface.
# Debian 13: the llama.cpp builds for the local VLM need glibc 2.38 or newer.
FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/reco-trainer
COPY local_worker.py ml_worker.py anonymize.py teachers.py augment_studio.py model_hub.py vlm.py model_registry.json package.json ./
COPY docker ./docker
COPY --from=ui /build/ui ./ui

ENV RECO_BIND_HOST=0.0.0.0
ENV RECO_VIDEO_FOLDER=/videos
EXPOSE 8766
ENTRYPOINT ["./docker/start-container.sh"]
