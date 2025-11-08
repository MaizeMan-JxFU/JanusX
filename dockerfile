FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update --quiet \
    && apt-get install --yes --quiet git \
    && apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends \
        python3 \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN git clone "https://gh-proxy.com/https://github.com/MaizeMan-JxFU/gtools.git" /app/gtools
ENV PATH="/app/gtools:$PATH"
WORKDIR /app/gtools

ENV UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && (python -m pip install . && cp doc/docker.sh gtools && chmod +x gtools && ./gtools gwas -h)

ENTRYPOINT ["gtools"]