.PHONY: all clean

all: $(patsubst %.ipynb, %.rst, $(wildcard *.ipynb))

clean:
	rm *.rst || true
	rm -r *_files/ || true
	rm -r ../source/*_files/ || true

%.rst: %.ipynb
	ipython nbconvert --to rst $<
	rm ../source/$@ || true
	rm -r ../source/$*_files || true
	cp $@ ../source/
	cp -r $*_files ../source/ || true
