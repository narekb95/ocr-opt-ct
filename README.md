HALG2PACE
=========
 A solver for the one-sided crossing minimization problem (OCM) parameterized by cutwidth.

* The solver assumes that a linear arrangement is given with input.
* Based on a dynamic programming approach over the given linear arragnement.
* Running time: $O(3^knk^c)$.

## Compilation and Running
The solver is given in a single `main.cpp` file and can be compiled as simple as
```
    g++ -std=c++11 main.cpp
```

A `Makefile` is provided for some compiling options:
- `make release`: Standard solver with standard input/output.
  - Example: `./release <[input-file] >[output-file]`.
- `make fre`: File input/output.
  - Syntax: `./fre [input-file] [output-file]`.
- `make out`: Outputs verbose data while running the solver, including cuts and dynamic programming states.
  - Syntax: `./out <[input-file] >[output-file]`.