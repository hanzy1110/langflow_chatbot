#!/bin/bash
# rm ./event_manager/migrations/*
docker-compose --env-file .env.prod down --remove-orphans
docker-compose --env-file .env.prod build
sleep 10
echo MAKING AND APPLYING MIGRATIONS...
# docker-compose --env-file .env.prod run --rm crm_api sh -c "python manage.py makemigrations && python manage.py migrate"
# docker-compose --env-file .env.prod run --rm crm_api sh -c "python manage.py makemigrations && python manage.py migrate"
# docker-compose --env-file .env.prod run --rm crm_api sh -c "python manage.py loaddata fixtures/fixtures.json"

# docker-compose --env-file .env.prod run --rm crm_api sh -c "python manage.py migrate"

docker-compose --env-file .env.prod up -d 

echo Waiting for containers...
sleep 10
docker ps -a 
docker logs -t weaviate

echo "----------------------<>-----------------------"
docker logs -t --follow chatbot_api

# docker exec -it crm_api bash 


