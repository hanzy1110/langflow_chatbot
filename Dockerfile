FROM python:3.10 AS builder-image

# create and activate virtual environment
# using final folder name to avoid path issues with packages
RUN python -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir -r requirements.txt

FROM python:3.10 AS runner-image
COPY --from=builder-image /home/venv /home/venv

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
