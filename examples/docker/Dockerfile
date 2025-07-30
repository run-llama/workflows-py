FROM python:3.13-alpine3.22

COPY . /opt/app
WORKDIR /opt/app

# Install system deps
RUN apk add --no-cache git

# Install workflows
RUN pip install -r requirements.txt

# Default configuration, override with "docker run -e NAME=value"
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080

CMD [ "python", "-m", "workflows.server", "app.py" ]
