ARG ENGINE_IMAGE
ARG TORCH_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3

FROM ${ENGINE_IMAGE} AS engine

FROM ${TORCH_IMAGE}
ARG KIT_REPO
ARG KIT_COMMIT
RUN git init -q /kit && git -C /kit fetch -q --depth 1 "$KIT_REPO" "$KIT_COMMIT" && git -C /kit checkout -q FETCH_HEAD \
    && git -C /kit submodule update -q --init --depth 1 vendor/TensorFold \
    && mkdir -p /kit/build/python-source \
    && git -C /kit/vendor/TensorFold archive HEAD | tar -x -C /kit/build/python-source \
    && for patch in /kit/patches/*.patch; do patch -s -d /kit/build/python-source -p1 < "$patch"; done
COPY --from=engine /opt/tensorfold /opt/tensorfold
ENV PATH=/opt/tensorfold/bin:$PATH \
    PY_SOURCE=/kit/build/python-source/src \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
WORKDIR /kit/engine/zig
