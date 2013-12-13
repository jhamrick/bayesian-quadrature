all: $(patsubst %.ipynb, %.rst, $(wildcard *.ipynb))

%.rst: %.ipynb
	ipython nbconvert --to rst $<
