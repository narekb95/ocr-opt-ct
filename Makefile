run: out
	./out < my-tests/1.graph
out: main.cpp
	g++ -D__DEBUG -std=c++14 -O2 -o out main.cpp