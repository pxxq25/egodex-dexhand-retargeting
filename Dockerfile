# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:5f0d2d827f6436b3cb7468fd8acbdc8c1d41261614e579ae49afe6141da51133

ENV DEBIAN_FRONTEND=noninteractive \
    EGODEX_DEXHAND_ROOT=/opt/egodex \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl ffmpeg git libgl1 libglib2.0-0 libvulkan1 \
      python3 python3-venv vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/egodex/project
COPY . /opt/egodex/project

# SAM3 is gated. Build with BuildKit and an environment-backed secret:
#   docker build --secret id=hf_token,env=HF_TOKEN -t egodex-dexhand .
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -f /run/secrets/hf_token ]; then \
      export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    fi; \
    ./bootstrap.sh --runtime /opt/egodex --skip-system-packages --python python3

ENTRYPOINT ["/opt/egodex/project/docker/entrypoint.sh"]
CMD ["bash"]
