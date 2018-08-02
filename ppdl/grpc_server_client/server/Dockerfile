FROM ubuntu:latest

COPY ./ /

# Python deps
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN pip install -r /requirements.txt

# Postgres deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y postgresql-10 postgresql-client-10 postgresql-contrib-10 

# allow local connections to DB
RUN cp /pg_hba.conf /etc/postgresql/10/main/pg_hba.conf

EXPOSE 1453

ENV DATABASE_URL=postgres://postgres@/?host=/var/run/postgresql

CMD ["/bin/bash", "-c", "service postgresql restart && cat schema.sql | psql $DATABASE_URL && python3 server.py"]
