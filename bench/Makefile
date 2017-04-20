gen: gen.cc
	g++ -std=c++14 -Wall -Wextra -g -fopenmp -o $@ $<

gen_b: gen.cc
	g++ -std=c++14 -Wall -Wextra -O -fopenmp -o $@ $<

bench: gen_b
	./$<

data.tsv: gen_b
	./$< > $@

plot.png: data.tsv plot.py
	python plot.py $< $@

clean:
	rm -f gen gen_b data.tsv 
