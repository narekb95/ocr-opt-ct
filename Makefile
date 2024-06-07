out: main.cpp
	g++ -D__DEBUG -std=c++14 -Wall -Werror -g -O0 -o out main.cpp

run: release
	./scripts/run_all_tiny.py
	
release: main.cpp
	g++ -std=c++14 -O2 -o release main.cpp