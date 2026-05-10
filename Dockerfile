FROM rust:1.83-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /home/app --shell /usr/sbin/nologin app

COPY --from=builder /app/target/release/me-api-proxy /usr/local/bin/me-api-proxy

ENV RUST_LOG=me_api_proxy=info,tower_http=info

EXPOSE 8080

USER app
WORKDIR /home/app

ENTRYPOINT ["/usr/local/bin/me-api-proxy"]
