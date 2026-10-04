.PHONY: .dat .hpl .pun .txt parser datafile parser-tests constants catalog data downloader catalog-support-tests reformed-tests

PYTEST ?= python -m pytest -q
PARSER_TESTS = testogsdat.py testogshpl.py testogspun.py testogstxt.py testogsdatafile.py testogsparser.py
CATALOG_SUPPORT_TESTS = testogsconstants.py testogscatalog.py testogsdata.py testogsdownloader.py

all: .dat .hpl .pun .txt parser datafile constants clustering downloader catalog data trainer

parser-tests: .dat .hpl .pun .txt datafile parser

catalog-support-tests: constants catalog data downloader

datafile:
	$(PYTEST) testogsdatafile.py

constants:
	$(PYTEST) testogsconstants.py

parser:
	$(PYTEST) testogsparser.py

.dat:
	$(PYTEST) testogsdat.py

.hpl:
	$(PYTEST) testogshpl.py

.pun:
	$(PYTEST) testogspun.py

.txt:
	$(PYTEST) testogstxt.py

clustering:
	python testogsclustering.py

downloader:
	$(PYTEST) testogsdownloader.py

catalog:
	$(PYTEST) testogscatalog.py

data:
	$(PYTEST) testogsdata.py

trainer:
	python testogstrainer.py
