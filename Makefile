IMAGE=rgpe
CONTAINER_NAME=rgpe
PWD=$(shell pwd)

build:
	docker build -t $(IMAGE) .

bash:
	docker exec -it $(CONTAINER_NAME) bash

start:
	docker run --rm -v "$(PWD)":/app -p 8888:8888 -d --name $(CONTAINER_NAME) $(IMAGE)

stop:
	docker stop $(CONTAINER_NAME)
