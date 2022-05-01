FROM nvcr.io/nvidia/tensorflow:22.03-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

WORKDIR /humbug

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /src ./src

CMD ["python", "src/predict.py"]