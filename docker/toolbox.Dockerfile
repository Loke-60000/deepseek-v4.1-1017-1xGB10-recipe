ARG ENGINE_IMAGE=ghcr.io/loke-60000/dsv41-1xgb10:latest
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
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli docker-buildx-plugin \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "huggingface_hub>=1.0" h5py scipy
COPY --from=engine /opt/tensorfold /opt/tensorfold

ARG ENGINE_IMAGE
ARG TOOLBOX_IMAGE=ghcr.io/loke-60000/dsv41-1xgb10-toolbox:latest
COPY pyproject.toml /recipe/
COPY bin /recipe/bin
COPY config /recipe/config
COPY docker /recipe/docker
COPY patches /recipe/patches
COPY recipe /recipe/recipe
RUN ln -s /recipe/bin/dsv41-cli /usr/local/bin/dsv41
ENV PATH=/opt/tensorfold/bin:$PATH \
    PY_SOURCE=/kit/build/python-source/src \
    PYTHONPATH=/recipe \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
    DSV41_IMAGE_TAG=${ENGINE_IMAGE} \
    DSV41_IMAGE_TOOLBOX_TAG=${TOOLBOX_IMAGE}
WORKDIR /kit/engine/zig
