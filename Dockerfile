#FROM funtoo/stage3-intel64-skylake
FROM rust:latest


RUN apt-get update && apt-get -y install openssl libclang-dev python3-dev python3-pip python3 python3-venv pipx

RUN mkdir /usr/target

RUN cd /usr; git clone https://github.com/ggerganov/llama.cpp.git; cd llama.cpp;make;

RUN pipx install 'transformers[torch]'
#RUN pip install 'transformers[tf-cpu]'
RUN pipx install joblib


WORKDIR /usr/workspace

#ENV RUSTFLAGS="--cfg tokio_unstable"
ENV CARGO_HOME=/usr/cargo_home
ENV CARGO_TARGET_DIR=/usr/target
ENV RUSTBERT_CACHE=/usr/rustbert_cache

RUN python3 -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"

CMD ["cargo build","cargo run"]
