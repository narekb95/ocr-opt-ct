out: main.cpp
	g++ -D__DEBUG -std=c++14 -Wall -Werror -g -O0 -o out main.cpp

run: out
	./out < my-tests/1.graph
	
release:
	g++ -std=c++14 -O2 -o out main.cpp