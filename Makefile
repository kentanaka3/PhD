Leo:
	bash Analyze.sh 1 4 MAKE src/Picker.py -D 230601 230605 -v --force
	bash Analyze.sh 1 2 MAKE src/Picker.py -D 230601 230605 -v --force
	bash Analyze.sh 1 1 MAKE src/Picker.py -D 230601 230605 -v --force

results:
	cp img/CP_EQTransformer.png doc/img/CP_EQTransformer.png
	cp img/CP_PhaseNet.png doc/img/CP_PhaseNet.png
	cp img/CM_* doc/img/
	cp img/TD_* doc/img/
	cp img/TPFN_* doc/img/
	cp img/TPFNFP_* doc/img/

testEnv:
	bash Analyze.sh 1 "1 2 4" TestEnv test/testEnv.py

testPicker:
	bash Analyze.sh 1 1 TestPicker test/testPicker.py
	bash Analyze.sh 1 "1 2 4" PickerTest src/Picker.py -d ./data/test/waveforms -v -D 230601 230604

testAnalyzer:
	bash Analyze.sh 1 1 TestAnalyzer test/testAnalyzer.py

testAssociator:
	bash Analyze.sh 1 "1 2 4" TestAssociator test/testAssociator.py

picker:
	bash Analyze.sh 1 "1 2 4" AUTOMATER src/Picker.py -v --force

analyzer:
	bash Analyze.sh 1 1 AUTOMATER src/Analyze.py -v --force --file ./data/manual/RSFVG-2023.dat

associator:
	bash Analyze.sh 1 "1 2 4" AUTOMATER src/Associator.py -v --force

clean:
	rm -f k*.err k*.out report*.nsys-rep report*.qdstrm; module purge; clear

clean_classify:
	rm -rf data/classified && clear