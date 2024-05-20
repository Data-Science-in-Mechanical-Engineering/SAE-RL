FROM nvidia/cuda:11.8.0-base-ubuntu20.04

FROM python:3.8

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN apt-get update
RUN apt-get install -y \
    git \
    xvfb x11-utils gnumeric
RUN apt-get clean

RUN pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv install --dev --system --deploy
RUN rm -f Pipfile Pipfile.lock

CMD [ "bash" ]