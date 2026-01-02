.PHONY: install train eval portfolio

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python scripts/train.py --config configs/default.yaml

eval:
	python scripts/eval.py --config configs/default.yaml

portfolio:
	python scripts/portfolio.py --config configs/default.yaml
