all: clean compile

compile: UniTs
.PHONY: compile

clean:
	@rm -rf *.aux *.log *.lot *.lof *.xml *.gz *.toc *.bcf *.fls *~
.PHONY: clean

MHPC: clean
	pdflatex MHPC.tex; pdflatex MHPC.tex
.PHONY: MHPC

UniTs: clean
	pdflatex UniTs.tex; pdflatex UniTs.tex
.PHONY: UniTs