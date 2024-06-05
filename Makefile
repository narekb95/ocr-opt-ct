run: out
	./out < data/1.gr
out: main.cpp
	g++ -std=c++14 -O2 -o out main.cpp