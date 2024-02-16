FROM rust:latest as core
RUN apt update && apt-get -y install openssl build-essential libc6 \
    libc-bin musl-tools libssl-dev pkg-config python3 \
    protobuf-compiler
WORKDIR /app
RUN cargo install cargo-chef
COPY . .
RUN cargo chef prepare --recipe-path recipe.json


FROM rust:latest as tester
RUN apt update && apt-get clean && apt-get -y -f install openssl build-essential libc6 \
    libc-bin musl-tools libssl-dev libvips-dev pkg-config python3 \
    protobuf-compiler supervisor llvm-dev libclang-dev clang
WORKDIR /app
RUN cargo install cargo-chef
COPY --from=core /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json
COPY . .