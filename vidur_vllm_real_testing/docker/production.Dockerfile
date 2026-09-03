ARG VLLM_IMAGE=vllm/vllm-openai:v0.13.0@sha256:d623253f2ba246378421c9642e20885e65257f38418ff26d48c81aea1702521b
FROM ${VLLM_IMAGE}

WORKDIR /opt/vidur
COPY . /opt/vidur/vidur_vllm_real_testing

ENV PYTHONPATH=/opt/vidur \
    VIDUR_PACKAGE_ROOT=/opt/vidur/vidur_vllm_real_testing \
    VIDUR_PINNED_TOKENIZER_REVISION=315b20096dc791d381d514deb5f8bd9c8d6d3061

RUN python3 -m vidur_vllm_real_testing.patch_vllm_scheduler install

ENTRYPOINT ["bash", "/opt/vidur/vidur_vllm_real_testing/docker/production_entrypoint.sh"]
CMD ["smoke-bundled"]
