FROM continuumio/miniconda3 AS builder-image

# avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive

# create and activate virtual environment
# using final folder name to avoid path issues with packages
# RUN python -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install -r requirements.txt

# FROM continuumio/miniconda3 AS runner_image
# COPY --from=builder-image /home/venv /home/venv

# copy project files
RUN mkdir /home/code
WORKDIR /home/code
COPY . .
RUN chmod +x entrypoint.sh

# Expose port
EXPOSE $API_PORT

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# activate virtual environment
ENV VIRTUAL_ENV=/home/venv
ENV PATH="/home/venv/bin:$PATH"
EXPOSE 4444
# CMD ["python manage.py runserver", "-D", "FOREGROUND"]
# using ubuntu LTS version
