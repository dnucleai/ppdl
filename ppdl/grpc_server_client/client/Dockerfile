FROM ubuntu:latest

COPY ./ /

# Python deps
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN pip install -r requirements.txt

CMD ["/bin/bash", "-c", "python3 client.py"]
