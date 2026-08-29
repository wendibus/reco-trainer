FROM node:22-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/reco-trainer
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

ENV RECO_BIND_HOST=0.0.0.0
ENV RECO_VIDEO_FOLDER=/videos
EXPOSE 8765 8766
ENTRYPOINT ["./docker/start-container.sh"]
