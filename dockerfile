FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update --quiet \
    && apt-get install --yes --quiet --no-install-recommends wget \
    && wget "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_25.9.1-1-Linux-x86_64.sh" --no-check-certificate \
    && bash Miniconda3-py312_25.9.1-1-Linux-x86_64.sh -b -p /app/miniconda \
    && rm Miniconda3-py312_25.9.1-1-Linux-x86_64.sh \
    && apt-get remove -y wget \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /app/JanusX
COPY . .
ENV PATH="/app/JanusX:/app/miniconda/bin:$PATH"

ENV UV_PYTHON_INSTALL_MIRROR="https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download"

RUN chmod +x ./install.sh; ./install.sh \
    && /app/miniconda/bin/conda clean -afy \
    && python -m uv cache clean && python -m pip cache purge \
    && ./jx gwas -h

ENTRYPOINT ["jx"]