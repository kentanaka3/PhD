downloader:
	python inc/downloader.py -v -N "*" -S "*" -D 240606 240620 #240416 240620

results:
	cp img/CP_EQTransformer_P.png doc/img/
	cp img/CP_EQTransformer_S.png doc/img/
	cp img/CP_PhaseNet_P.png doc/img/
	cp img/CP_PhaseNet_S.png doc/img/
	cp img/CM_* doc/img/
	cp img/TD_* doc/img/
	cp img/TPFN_* doc/img/
	cp img/TPFNFP_* doc/img/

training:
	python src/picker.py -v --train

testing:
	python test/testparser.py
	python test/testanalyzer.py
	python test/testinitializer.py
	python test/testpicker.py
	python test/testassociator.py

testEnv:
	python test/testEnv.py

testPicker:
	python test/testPicker.py
	python src/picker.py -d ./data/test/waveforms -v -D 230601 230604

testAnalyzer:
	python test/testAnalyzer.py
	python src/analyzer.py -v --file ./data/test/manual -d ./data/test/waveforms -D 230601 230604

testAssociator:
	python test/testassociator.py
	python src/associator.py -v -d ./data/test/waveforms -D 230601 230604

testParser:
	python test/testparser.py

trainer:
	python src/trainer.py --file ./data/manual/ -v -D 230601 230610 -W OGS
	python src/trainer.py --file ./data/manual/ -v -D 230101 230610 -W OGS23
	python src/trainer.py --file ./data/manual/ -v -D 200101 201231 -W OGS20

picker:
	python src/picker.py -v -D 230601 230610

analyzer:
	python src/analyzer.py -v --file ./data/manual -D 230601 230610

associator:
	python src/associator.py -v -D 230601 230610

pipeline: picker associator analyzer

clean_classify:
	rm -rf data/classified && clear
