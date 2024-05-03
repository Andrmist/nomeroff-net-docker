FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#RUN apt-key del 7fa2af80 && \
#    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt update && apt install -y git libsm6 libxrender1 libgl1 libfontconfig1 libxtst6 build-essential libglib2.0-0

RUN pip install --no-cache-dir flask waitress torch wtforms

ARG NOMEROFF_NET_REV

WORKDIR /var/www
RUN git clone https://github.com/ria-com/nomeroff-net.git \
  && cd nomeroff-net/ \
  && git checkout $NOMEROFF_NET_REV \
  && sed -i 's/^tensorflow/#&/' requirements.txt \
  && pip install --no-cache-dir -r requirements.txt
