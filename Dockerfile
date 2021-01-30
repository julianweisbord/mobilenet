FROM tensorflow/tensorflow:latest
WORKDIR /
COPY ./mobilenet.py /
CMD python3 mobilenet.py
