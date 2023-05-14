#!/bin/bash
# rm ./event_manager/migrations/*
docker-compose --env-file .env.dev down --remove-orphans
docker-compose --env-file .env.dev build
sleep 10
docker-compose --env-file .env.dev up -d 

echo Waiting for containers...
sleep 10
docker ps -a 
docker logs -tf weaviate

# echo "----------------------<>-----------------------"
# docker logs -t --follow chatbot_api

# docker exec -it crm_api bash 


