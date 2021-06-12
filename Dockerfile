FROM python:3.8

ADD requirements.txt .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["uvicorn", "api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
