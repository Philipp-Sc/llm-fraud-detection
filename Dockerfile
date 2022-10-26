#FROM funtoo/stage3-intel64-skylake
FROM rust:latest

RUN mkdir /usr/target

WORKDIR /usr/workspace


CMD ["cargo build","cargo run"]
