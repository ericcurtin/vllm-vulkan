FROM registry.fedoraproject.org/fedora:44 AS builder

RUN dnf install -y \
        python3.12 \
        python3.12-devel \
        rust \
        cargo \
        vulkan-loader-devel \
        vulkan-headers \
        glslang \
        gcc \
        g++ \
        cmake \
        git \
        numactl-devel \
        curl \
        pkg-config \
    && dnf clean all

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/vllm-vulkan
RUN uv venv "$VIRTUAL_ENV" --python 3.12 --seed
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY .vllm-version /tmp/.vllm-version
RUN VLLM_VERSION=$(cat /tmp/.vllm-version) \
    && curl -fsSL "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}.tar.gz" \
        -o /tmp/vllm.tar.gz \
    && tar xf /tmp/vllm.tar.gz -C /tmp \
    && uv pip install -r "/tmp/vllm-${VLLM_VERSION}/requirements/cpu.txt" \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && VLLM_TARGET_DEVICE=cpu CXXFLAGS="-Wno-parentheses" uv pip install "/tmp/vllm-${VLLM_VERSION}" \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /tmp/vllm*

WORKDIR /build
COPY . .
RUN uv pip install .


FROM registry.fedoraproject.org/fedora:44

RUN dnf install -y \
        python3.12 \
        vulkan-loader \
        mesa-vulkan-drivers \
    && dnf clean all

COPY --from=builder /opt/vllm-vulkan /opt/vllm-vulkan
ENV VIRTUAL_ENV=/opt/vllm-vulkan \
    PATH="/opt/vllm-vulkan/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
