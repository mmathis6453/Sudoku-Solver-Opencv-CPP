#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <fstream>
#include <string>

using namespace std;

vector<int>** copyMoveset(vector<int>** moveset){

	vector<int>** newMoveset;
	newMoveset = new vector<int>*[9];
	for (int i=0; i<9; i++){
		newMoveset[i] = new vector<int>[9];
	}
	for (int i=0; i<9; i++){
		for (int j=0; j<9; j++){
			newMoveset[i][j] = moveset[i][j];
		}
	}
	return newMoveset;
}

bool resolveMoveset(vector<int>** moveset, int i, int j, int num){

	vector<int>::iterator it;
	for (int k=0; k<9; k++){
		// Resolve row
		if (k!=j){
			it = find(moveset[i][k].begin(), moveset[i][k].end(), num);
			if (it != moveset[i][k].end()){
				it = moveset[i][k].erase(it);
				if (moveset[i][k].empty()){
					return false;
				}
			}
		}
		// Resolve column
		if (k!=i){
			it = find(moveset[k][j].begin(), moveset[k][j].end(), num);
			if (it != moveset[k][j].end()){
				it = moveset[k][j].erase(it);
				if (moveset[j][j].empty()){
					return false;
				}
			}
		}
	}

	// Resolve sector
	int i2 = (i/3*3);
	int j2 = (j/3*3);
	for (int m=i2; m<i2+3; m++){
		for (int n=j2; n<j2+3; n++){
			if (m!=i && n!=j){
				it = find(moveset[m][n].begin(), moveset[m][n].end(), num);
				if (it != moveset[m][n].end()){
					it = moveset[m][n].erase(it);
					if (moveset[m][n].empty()){
						return false;
					}
				}
			}
		}
	}
	// Set move
	vector<int> v;
	v.push_back(num);
	moveset[i][j] = v;
	return true;
	
}

bool checkWin(vector<int>** moveset){
	for (int i=0; i<9; i++){
		for (int j=0; j<9; j++){
			if (moveset[i][j].size() != 1){
				return false;
			}
		}
	}
	return true;
}

void printGame(vector<int>** moveset){
	for (int i=0; i<9; i++){
		for (int j=0; j<9; j++){
			cout<<moveset[i][j][0]<<" ";
		}
		cout<<endl;
	}
}

bool solveGame(int game[][9]){
	// Create the initial moveset
	vector<int>** moveset;

	moveset = new vector<int>*[9];
	for (int i=0; i<9; i++){
		moveset[i] = new vector<int>[9];
	}

	int ints[] = {1,2,3,4,5,6,7,8,9};
	for (int i=0; i<9; i++){
		for (int j=0; j<9; j++){
			vector<int> v(ints, ints + sizeof ints / sizeof ints[0]);
			moveset[i][j] = v;
		}
	}

	// Resolve initial moveset with initial game
	for (int i=0; i<9; i++){
		for (int j=0; j<9; j++){
			if (game[i][j] != 0){

				if (!resolveMoveset(moveset, i, j, game[i][j])){
					cout<<"Invlid game"<<endl;
					return false;
				}
			}
		}
	}

	// Run logic
	stack<vector<int>**> gameStack;
	gameStack.push(moveset);

	int i=0;
	int j=0;

	while(true){

		// If we win, set values and exit
		if (checkWin(gameStack.top())){
			//printGame(gameStack.top());
			for (int i=0; i<9; i++){
				for (int j=0; j<9; j++){
					game[i][j] = gameStack.top()[i][j][0];
				}
			}
			return true;
		}

		// If not more moves for the cell, backtrack
		if (gameStack.top()[i][j].empty()){
			gameStack.pop();
			if(gameStack.empty()){
				cout<<"Couldn't solve game. Make sure a valid game was passed in"<<endl;
				return false;
			}
			i--;
			if (i<0){
				i=8;
				j--;
			}
		}

		else {
			// Copy the top level moveset and try a move
			vector<int>** newMoveset = copyMoveset(gameStack.top());

			// Remove the move fro the movelist
			int num = gameStack.top()[i][j].back();
			gameStack.top()[i][j].pop_back();

			// If the move resolves, continue on
			if (resolveMoveset(newMoveset, i, j, num)){
				gameStack.push(newMoveset);
				i++;
				if (i==9){
					i=0;
					j++;
				}
			}
		}
	}
}
