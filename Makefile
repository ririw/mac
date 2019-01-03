all: lint test check

lint:
	flake8 src/mac
	flake8 tests

test:
	nosetests

check:
	mac-learn check


sync:
	rsync --exclude tests/__pycache__ \
		  --exclude src/mac/__pycache__ \
		  --exclude .git \
		  --exclude docs \
		  --exclude .mypy_cache \
		  --exclude .idea \
		  --exclude .cache \
		  -r --progress -a \
		  . riri@learnbox:~/Documents/mac