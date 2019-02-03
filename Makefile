all: lint test check

lint:
	flake8 src/mac
	flake8 tests
	flake8 decompress.py

test:
	nosetests

check:
	mac-learn check

docker-base:
	docker build -t registry.gitlab.com/ririw/mac/base - < Dockerfile-base

docker: docker-base
	docker build -t registry.gitlab.com/ririw/mac .

docker-push: docker docker-base
	docker push registry.gitlab.com/ririw/mac/base
	docker push registry.gitlab.com/ririw/mac

sync:
	rsync --exclude tests/__pycache__ \
		  --exclude src/mac/__pycache__ \
		  --exclude .git \
		  --exclude docs \
		  --exclude .mypy_cache \
		  --exclude .idea \
		  --exclude .cache \
		  --exclude .ipynb_checkpoints \
		  -r --progress -a \
		  . riri@learnbox.local:~/Documents/mac
