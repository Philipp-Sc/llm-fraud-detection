#FROM funtoo/stage3-intel64-skylake
FROM rust:latest

RUN mkdir /usr/target

WORKDIR /usr/workspace

#ENV RUSTFLAGS="--cfg tokio_unstable"
ENV CARGO_HOME=/usr/cargo_home
ENV CARGO_TARGET_DIR=/usr/target
ENV RUSTBERT_CACHE=/usr/rustbert_cache

CMD ["cargo build","cargo run"]
