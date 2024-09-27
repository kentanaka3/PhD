all:
	python -m src.AdriaArray

Leo:
	bash Analyze.sh src/AdriaArray.py

results:
	cp img/CP_EQTransformer.png doc/img/CP_EQTransformer.png
	cp img/CP_PhaseNet.png doc/img/CP_PhaseNet.png
	cp img/CM_* doc/img/
	cp img/TD_* doc/img/
	cp img/TPFN_* doc/img/

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