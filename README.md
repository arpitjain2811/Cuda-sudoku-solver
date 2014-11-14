cuda-sudoku-solver
==================
### Algorithm
Sudoku is solved using simulated annealing

1. The program accepts a partially completed sudoku grid, where 0 represent an empty spot.

2. All the 9 3x3 sub grids are filled with numbers which are feasible for that grid ie. this random filling will take care of the one of the three constraints of the problem. Thus all the 9 sub grids will have 1-9 exactly once.

3. Next a loop is run until the minimum temperature is achieved.
  1. Select a random sub grid.

  2. Select two random mutable points ( ie. 0 in given sudoku ) randomly.

  3. Swap the values of these randomly selected points.

  4. Calculate the energy or fitness. The fitness function used here is: (81*2) - ( total number of unique numbers in each row and column ). When the sudoku is solved this energy will be zero.

  5. Accept the new state if its energy is lower than the previous state.

  6. Else still accept or reject the new state with the acceptance probability.

4. Decrease the temperature according to the cooling schedule ( current_temperture * 0.8 ).


### Compilation and Running
`make`

`./sudoku_solver test_input.in`

Where test_input.in is a file containing a sudoku problem.
