all:
	python src/AdriaArray.py -G BEGDT NETWORK STATION -v
testing:
	python -m test.test_AdriaArray

clean_annotate:
	rm -rf data/annotated

clean_classify: clean_annotate
	rm -rf data/classified

clean_process: clean_classify
	rm -rf data/processed