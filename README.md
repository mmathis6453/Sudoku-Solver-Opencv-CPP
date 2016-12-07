# Sudoku-Solver-Opencv-CPP
Parses image of sudoku game, and prints solution on the image

This is an extention of my other repository that takes in a sudoku game via text, and outputs the solved version.
It uses opencv 2.4 to parse an image of a sudoku game, and uses OCR to convert the image to text. The program then passes the text into the solver algorithm, and prints the resulting solution on the image.

##To Build
To build this, you will need opencv 2.4 installed on your computer (I haven't tested it with other versions of opencv). You can then use your favorite compiler to link the program to opencv to generate the executable. 

For those who don't want to go through the trouble of installing opencv, I've compiled this using Visual Studio 2010 and included the windows executable in the resository. 

I plan on adding a cmake for this project in the future. 

##To Run
The program can be run from the command line, using the image of the sudoku puzzle as the first argument. If the program runs successfully, a file will be saved with a "solved" added to the filename. Any errors will be printed to the console. 

Example: solve_sudoku.exe sudoku_game.jpg

##Things That Can Go Wrong
This program has trouble with some images. The two biggest issues I've encountered are when the resolution of the image is too low, and if the cells in the puzzle are not fully enclosed. 

Sometimes the program will get the digit in the cell wrong. This will often result in the algorithm not being able to solve the puzzle. If this occurs, the program will print out what it thinks the numbers in the puzzle are to the console. When this occurs, you can have the program learn new digits from the image.
To do this, run the program with "learn" as the second argument. Note that this does require windowing so it will not work via ssh.

Example: solve_sudoku.exe sudoku_game.jpg learn

This will pop up a series of windows with a character on it. Type the digit you see and the image will be saved for the program to use. Once this is completed, the program should parse the game correctly. 
