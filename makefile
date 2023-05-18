test:
	PYTHONPATH=./src pytest tests

build-package:
	rm -rf dist/
	rm -rf *.egg-info
	python -m build


publish:
	python -m twine upload --repository pypi dist/*
