.DEFAULT_GOAL := help
.PHONY: help setup install start stop

help:
	@printf '%s\n' 'make setup   Create the Python environment and local directories' 'make install Install frontend and pinned GPU backend dependencies' 'make start   Start both services in the background' 'make stop    Stop services started by make start'

setup:
	@./setup.sh

install:
	@./install.sh

start:
	@./start.sh

stop:
	@./stop.sh
