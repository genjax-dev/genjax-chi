FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
MAINTAINER McCoy "Hoss" Becker <mccoyb@mit.edu>

RUN apt-get update && apt-get install -y --no-install-recommends \
    virtualenv \
    curl \
    pipx \
    git \
    ranger \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

RUN groupadd -g 1001 wizard
RUN useradd -rm -d /home/wizard -s /bin/bash -g wizard -G sudo -u 1001 wizard
USER wizard
WORKDIR /home/wizard

RUN pipx ensurepath
RUN pipx install poetry

SHELL ["/bin/bash", "-l", "-c"]
ADD . /home/wizard
