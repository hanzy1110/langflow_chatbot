#!/bin/sh
uvicorn main:app --reload --host 0.0.0.0 --port ${API_PORT} 
