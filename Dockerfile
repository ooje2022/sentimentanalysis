# Set base image (host OS)
FROM python:3.8-alpine
USER root
WORKDIR /app
ADD . /app
RUN apt update && apt install --no-install-recommends -y python3-dev  gcc build-essential
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]