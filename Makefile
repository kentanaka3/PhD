Leo:
	bash Analyze.sh 1 4 MAKE src/picker.py -D 230601 230605 -v --force
	bash Analyze.sh 1 2 MAKE src/picker.py -D 230601 230605 -v --force
	bash Analyze.sh 1 1 MAKE src/picker.py -D 230601 230605 -v --force

results:
	cp img/CP_EQTransformer_P.png doc/img/
	cp img/CP_EQTransformer_S.png doc/img/
	cp img/CP_PhaseNet_P.png doc/img/
	cp img/CP_PhaseNet_S.png doc/img/
	cp img/CM_* doc/img/
	cp img/TD_* doc/img/
	cp img/TPFN_* doc/img/
	cp img/TPFNFP_* doc/img/

testing:
	python test/testparser.py
	python test/testanalyzer.py
	python test/testinitializer.py
	python test/testpicker.py
	python test/testassociator.py

testEnv:
	bash Analyze.sh 1 "1 2 4" TestEnv test/testEnv.py

testPicker:
	bash Analyze.sh 1 1 TestPicker test/testPicker.py
	bash Analyze.sh 1 "1 2 4" PickerTest src/picker.py -d ./data/test/waveforms -v -D 230601 230604

testAnalyzer:
	bash Analyze.sh 1 1 TestAnalyzer test/testAnalyzer.py

testAssociator:
	bash Analyze.sh 1 "1 2 4" TestAssociator test/testassociator.py

testParser:
	python test/testparser.py

picker:
	bash Analyze.sh 1 4 AUTOMATER_PICKER src/picker.py -v -D 230601 240630

analyzer:
	bash Analyze.sh 1 1 AUTOMATER_ANALYZER src/analyzer.py -v --file ./data/manual -D 230601 240630

associator:
	bash Analyze.sh 1 1 AUTOMATER_ASSOCIATOR src/associator.py -v

clean:
	rm -f k*.err k*.out report*.nsys-rep report*.qdstrm; module purge; clear

clean_classify:
	rm -rf data/classified && clear