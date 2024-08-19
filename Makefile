all:
	python -m src.AdriaArray

Leo:
	bash Analyze.sh src/AdriaArray.py

testing:
	python -m test.testAdriaArray

clean:
	rm -f K*.err K*.out; clear

clean_annotate:
	rm -rf data/annotated && clear

clean_classify: clean_annotate
	rm -rf data/classified && clear

clean_process: clean_classify
	rm -rf data/processed && clear