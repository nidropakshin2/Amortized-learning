all: build clean

build:
	latexmk -pdf -cd -out2dir=.. -time -verbose src/main.tex
clean: build
# 	latexmk -cd -c src/main.tex
	rm src/*.pdf 