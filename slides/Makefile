OUT_NAME = Flow Matching

all: build move

build:
	latexmk -pdf -cd -time -verbose -f src/main.tex
move: build
	mv src/main.pdf "./$(OUT_NAME).pdf"
clean: 
	latexmk -cd -c src/main.tex 