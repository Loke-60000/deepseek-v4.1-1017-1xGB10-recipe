ARG CUDA_DEVEL=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG CUDA_RUNTIME=nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04

FROM alpine:3.20 AS source
ARG KIT_REPO
ARG KIT_COMMIT
RUN apk add --no-cache git && git init -q /kit && git -C /kit fetch -q --depth 1 "$KIT_REPO" "$KIT_COMMIT" && git -C /kit checkout -q FETCH_HEAD \
    && mv /kit/engine/zig /src
COPY patches/ /patches/
RUN cd /src && for patch in /patches/*.patch; do [ -e "$patch" ] || continue; git apply --verbose "$patch"; done

FROM ${CUDA_DEVEL} AS build
ARG ZIG_VERSION=0.17.0
ARG ZIG_TARGET=aarch64-linux-gnu
ARG ZIG_CPU=baseline+sha2+aes
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/zig \
    && curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" \
       | tar -xJ -C /opt/zig --strip-components=1
ENV PATH=/opt/zig:/usr/local/cuda/bin:$PATH
COPY --from=source /src /src
WORKDIR /src
RUN nvcc -fatbin -O3 -std=c++17 \
        -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 \
        -gencode=arch=compute_90,code=compute_90 \
        -o /tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu \
    && zig build -Dtarget="$ZIG_TARGET" -Dcpu="$ZIG_CPU" -Dnvcc=/usr/local/cuda/bin/nvcc \
        -Dtp-fatbin=/tp_mailbox.fatbin -Dsm=121 -Doptimize=ReleaseSafe -j4 --prefix /out install \
    && test -x /out/bin/tensorfold-dsv41 && test -x /out/bin/tf-dsv41-m1

FROM ${CUDA_RUNTIME}
ARG KIT_COMMIT
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libibverbs1 ibverbs-providers rdma-core \
        curl ca-certificates libcublas-13-3=13.6.0.2-1 \
    && apt-mark hold libcublas-13-3 \
    && dirname "$(dpkg -L libcublas-13-3 | grep '/libcublas\.so\.13$')" > /etc/ld.so.conf.d/990_cublas-13-3.conf \
    && ldconfig && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/tensorfold-dsv41 /out/bin/tf-dsv41-m1 /opt/tensorfold/bin/
COPY --from=build /tp_mailbox.fatbin /opt/tensorfold/share/tp_mailbox.fatbin
COPY docker/libproof.sh /opt/tensorfold/share/libproof.sh
RUN bash /opt/tensorfold/share/libproof.sh
ENV PATH=/opt/tensorfold/bin:$PATH \
    TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin
LABEL org.tensorfold.kit-commit="${KIT_COMMIT}"
