.PHONY: test reproduce figures paper
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

test:
	python -m pytest -q
reproduce: test
	python experiments/run_all.py --suite all
	python experiments/stress_controls.py
	python experiments/run_all.py --suite summary
	python experiments/plot_results.py
	python scripts/verify_results.py
figures:
	python experiments/plot_results.py
paper:
	bash scripts/build_paper.sh
