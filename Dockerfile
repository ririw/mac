FROM python:3.6

RUN pip3 install plumbum torch nose
RUN pip3 install nose numpy scipy flake8
RUN mkdir /app
RUN mkdir /data

COPY . /app
WORKDIR /app
RUN python setup.py develop
RUN make
