FROM rust:latest

RUN apt-get update && apt-get -y install openssl libclang-dev

RUN mkdir /usr/target

RUN cd /usr; git clone https://github.com/ggerganov/llama.cpp.git; cd llama.cpp;make;

WORKDIR /usr/workspace

ENV CARGO_HOME=/usr/cargo_home
ENV CARGO_TARGET_DIR=/usr/target
ENV RUSTBERT_CACHE=/usr/rustbert_cache

CMD ["cargo build","cargo run"]
