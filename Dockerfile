FROM jhonatans01/python-dlib-opencv
COPY . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "main:app"]