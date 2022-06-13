FROM docker pull nvcr.io/nvidia/pytorch:22.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

WORKDIR /humbug

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /src ./src

CMD ["python", "src/predict.py"]