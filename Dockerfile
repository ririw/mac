FROM python:3.6

# These lines install the minimum required to
# run the data-download script. By doing this
# early, hopefully we avoid running this very
# slow couple of lines very often, because it
# would start to get a bit frustrating if you
# had to re-download for every build and test
# cycle.
RUN pip3 install plumbum torch
RUN mkdir /app
RUN mkdir /data

COPY . /app
WORKDIR /app
RUN python setup.py develop
RUN
