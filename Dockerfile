FROM registry.gitlab.com/ririw/mac/base

RUN mkdir /app
RUN mkdir /data

COPY . /app
WORKDIR /app
RUN python setup.py develop
RUN make
