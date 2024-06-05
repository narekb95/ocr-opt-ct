run: out
	./out < mini-test.in
out: main.cpp
	g++ -std=c++14 -O2 -o out main.cpp