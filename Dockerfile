FROM nvidia/cuda:11.5.2-runtime-ubuntu20.04

WORKDIR /app
COPY . /app

RUN apt-get update && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -e '.[tests,docs,build]' && \
    pip3 install --no-cache-dir jupyter

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]