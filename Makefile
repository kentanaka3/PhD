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

testLcl:
	python test/testparser.py
	python test/testanalyzer.py
	python test/testinitializer.py
	python test/testpicker.py

testEnv:
	bash Analyze.sh 1 "1 2 4" TestEnv test/testEnv.py

testPicker:
	bash Analyze.sh 1 1 TestPicker test/testPicker.py
	bash Analyze.sh 1 "1 2 4" PickerTest src/picker.py -d ./data/test/waveforms -v -D 230601 230604

testAnalyzer:
	bash Analyze.sh 1 1 TestAnalyzer test/testAnalyzer.py

testAssociator:
	bash Analyze.sh 1 "1 2 4" TestAssociator test/testAssociator.py

testParser:
	python test/testparser.py

picker:
	bash Analyze.sh 1 "1 2 4" AUTOMATER src/picker.py -v --force

analyzer:
	bash Analyze.sh 1 1 AUTOMATER src/analyzer.py -v --file ./data/manual/RSFVG-2023.dat -D 230601 240630

associator:
	bash Analyze.sh 1 "1 2 4" AUTOMATER src/catalogger.py -v --force

clean:
	rm -f k*.err k*.out report*.nsys-rep report*.qdstrm; module purge; clear

clean_classify:
	rm -rf data/classified && clear