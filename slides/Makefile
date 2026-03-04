OUT_NAME = CV

all: build clean

build:
	latexmk -pdf -cd -time -verbose src/main.tex
clean: build
# 	latexmk -cd -c $(PAPER_PATH)/src/main.tex 
	mv src/main.pdf "./$(OUT_NAME).pdf"