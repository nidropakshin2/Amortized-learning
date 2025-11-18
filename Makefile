PAPER_PATH = paper

all: build clean

build:
	latexmk -pdf -cd -time -verbose $(PAPER_PATH)/src/main.tex
clean: build
# 	latexmk -cd -c $(PAPER_PATH)/src/main.tex 
	mv $(PAPER_PATH)/src/main.pdf $(PAPER_PATH)/main.pdf