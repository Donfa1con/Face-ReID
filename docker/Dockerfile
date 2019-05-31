FROM ubuntu:16.04

ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/15461/l_openvino_toolkit_p_2019.1.133.tgz

ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer

RUN apt-get update && apt-get install -y \
	curl \
	git \
	python3-pip \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $TEMP_DIR && \
    cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR && \
    $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh

WORKDIR /code

COPY requirements.txt  /code/
COPY docker/model.lst /code/docker/

RUN pip3 install -r requirements.txt

RUN /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --list docker/model.lst --output models

ENV FACE_MODEL_PATH=./models/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001 \
    REID_MODEL_PATH=./models/Retail/object_reidentification/face/mobilenet_based/dldt/face-reidentification-retail-0095 \
    PLUGIN_PATH=/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx2.so

COPY . /code

ENTRYPOINT ["/bin/bash", "docker/entrypoint.sh"]