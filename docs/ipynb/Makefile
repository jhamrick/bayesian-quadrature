.PHONY: all clean

all: $(patsubst %.ipynb, %.rst, $(wildcard *.ipynb))

clean:
	rm -f *.rst
	rm -rf *_files/
	rm -rf ../source/*_files/

%.rst: %.ipynb
	ipython nbconvert --to rst $<
	rm -f ../source/$@
	rm -rf ../source/$*_files
	cp $@ ../source/
	cp -r $*_files ../source/ || true
