# COMMON
FROM ubuntu:22.04 AS app_base
# Pre-reqs
RUN apt-get update && apt-get install --no-install-recommends -y \
    git vim build-essential python3-dev python3-venv python3-pip
# Instantiate venv and pre-activate
RUN pip3 install virtualenv
RUN virtualenv /venv
# Credit, Itamar Turner-Trauring: https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install --upgrade pip setuptools
# Copy and enable all scripts
COPY ./scripts /scripts
RUN chmod +x /scripts/*
### DEVELOPERS/ADVANCED USERS ###
# Clone text-generation-webui
RUN git clone https://github.com/Vizxe/text-generation-webui /src

RUN cp -ar /src /app

FROM app_base AS app_nvidia
# Install pytorch for CUDA 12.1
RUN pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121 
# Install oobabooga/text-generation-webui
RUN pip3 install -r /app/requirements.txt

# COMMON
FROM ubuntu:22.04 AS run_base
# Runtime pre-reqs
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-venv python3-dev git
# Copy app and src
COPY --from=app_base /app /app
COPY --from=app_base /src /src
# Instantiate venv and pre-activate
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Finalise app setup
WORKDIR /app
EXPOSE 7860
# Required for Python print statements to appear in logs
ENV PYTHONUNBUFFERED=1
# Force variant layers to sync cache by setting --build-arg BUILD_DATE
ARG BUILD_DATE
ENV BUILD_DATE=$BUILD_DATE
RUN echo "$BUILD_DATE" > /build_date.txt
# Copy and enable all scripts
COPY ./scripts /scripts
RUN chmod +x /scripts/*
# Run
ENTRYPOINT ["/scripts/docker-entrypoint.sh"]


FROM run_base AS base-nvidia
# Copy venv
COPY --from=app_nvidia $VIRTUAL_ENV $VIRTUAL_ENV
# Variant parameters
RUN echo "Nvidia Base" > /variant.txt
ENV EXTRA_LAUNCH_ARGS=""
CMD ["python3", "/app/server.py"]