# Makefile for Fraud Detection API

# Variables
IMAGE_NAME=fraud-api
CONTAINER_NAME=fraud-api-container
PORT=8000

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the container
run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):8000 $(IMAGE_NAME)

# Stop and remove the container
stop:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

# Rebuild (clean + build)
rebuild: stop build run

# Tail logs from container
logs:
	docker logs -f $(CONTAINER_NAME)

# Open shell inside the container
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash