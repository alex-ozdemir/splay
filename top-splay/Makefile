top-splay: top-splay.cc
	g++ -std=c++14 -Wall -Wextra -g -o $@ $<

plot.png: plot.py data.tsv
	python $< data.tsv $@

clean:
	rm -f plot.png
