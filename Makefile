all: lint test check

lint:
	flake8 src/mac
	flake8 tests

test:
	nosetests

check:
	mac-learn check