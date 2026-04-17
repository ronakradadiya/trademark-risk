FROM node:24-slim AS builder
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 build-essential ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /src/app
COPY app/package.json app/package-lock.json ./
RUN npm ci
COPY app ./
RUN npm run build

FROM node:24-slim AS runner
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    USPTO_DB=/data/uspto.sqlite \
    DYNAMO_AUDIT_DISABLED=1

WORKDIR /app
COPY --from=builder /src/app/.next ./app/.next
COPY --from=builder /src/app/node_modules ./app/node_modules
COPY --from=builder /src/app/package.json ./app/package.json
COPY --from=builder /src/app/next.config.mjs ./app/next.config.mjs
COPY ml/models ./ml/models

WORKDIR /app/app
EXPOSE 3000
CMD ["npm", "start"]
