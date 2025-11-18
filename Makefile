.PHONY: build-docker run-docker clean-docker

DOCKER_IMAGE ?= pc-rwalker-dev
DOCKER_FILE ?= dockerfile
DOCKER_RUN_FLAGS ?= --rm -it -v $(PWD):/workspace -w /workspace

build-docker:
	docker build -t $(DOCKER_IMAGE) -f $(DOCKER_FILE) .

run-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE)

clean-docker:
	- docker rm -f $$(docker ps -aq --filter ancestor=$(DOCKER_IMAGE)) >/dev/null 2>&1 || true
	- docker rmi -f $(DOCKER_IMAGE) >/dev/null 2>&1 || true
	- docker image prune -f >/dev/null 2>&1 || true
