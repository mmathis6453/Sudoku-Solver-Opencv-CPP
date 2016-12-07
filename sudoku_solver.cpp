#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <sstream>

using namespace cv;
using namespace std;

bool solveGame(int game[][9]);

int fileCounts[] = {0,0,0,0,0,0,0,0,0,0};

int sample_width = 15;
int sample_height = 15;
int sample_area = sample_width * sample_height;

//Figures out the amount of learn files for each charcter
void setFileCounts(){
	char readFile[256];
	for (unsigned i=1; i<=9; i++){
		unsigned j = 0;
		while (true){
			sprintf(readFile, "./learn/%d_%d.jpg", i, j);
			Mat read_img = imread(readFile, CV_32FC1);
			j++;
			if (!read_img.data){
				break;
			}
			fileCounts[i]++;
		}
	}
}

//Saves the learn files
void save_learned(Mat &img, char c){
	ostringstream os;
	os<<"learn/";
	int i = c-'0';
	os<<c<<"_"<<fileCounts[i]<<".jpg";
	imwrite(os.str(), img);
	fileCounts[i]++;
}

//Returns cosine of angle
double angle( Point pt1, Point pt2, Point pt0 ){
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// This was adapted from a tutorial on square recognition 
void findSquares(const Mat& image, vector<vector<Point> >& squares , int thresh_val){
    squares.clear();

    vector<vector<Point> > contours;

	Mat gray, gaus, thresh, canny;

	cvtColor(image, gray, CV_BGR2GRAY);
	GaussianBlur(gray,gaus,Size(3,3), 0, 0);
	threshold(gaus, thresh, thresh_val, 255, THRESH_BINARY);

    findContours(thresh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    vector<Point> approx;

    for( size_t i = 0; i < contours.size(); i++ ){
        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

        if( approx.size() == 4 &&
			fabs(contourArea(Mat(approx))) > image.size().area()/500 &&
			fabs(contourArea(Mat(approx))) < image.size().area()/50 &&
            isContourConvex(Mat(approx)) ){

            double maxCosine = 0;

            for( int j = 2; j < 5; j++ ){
                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }
            if( maxCosine < 0.3 )
                squares.push_back(approx);
        }
    }
}

//Comparative functions for sorting
bool sort_x(Rect a, Rect b){
	return a.x < b.x;
}

bool sort_y(Rect a, Rect b){
	return a.y < b.y;
}

bool sort_bound_area(vector<Point> a, vector<Point> b){
	return boundingRect(Mat(a)).area() > boundingRect(Mat(b)).area();
}

//Takes the squares returned from the findSquares function, converts them to rects, 
//and sorts them based on location. Left to right, top to bottom, like a book
void sortConvertSquares(const vector<vector<Point> >& squares, vector<Rect>& orderedRects){
	for( size_t i = 0; i < squares.size(); i++){
		int compare_max = 0; 
		int compare_min = INT_MAX;
		Point top_left;
		Point bottom_right;
		for( size_t j = 0; j < squares[i].size(); j++){
			if (squares[i][j].x+squares[i][j].y < compare_min){
				compare_min = squares[i][j].x+squares[i][j].y;
				top_left = squares[i][j];
			}
			if (squares[i][j].x+squares[i][j].y > compare_max){
				compare_max = squares[i][j].x+squares[i][j].y;
				bottom_right = squares[i][j];
			}
		}

		Rect r(top_left, bottom_right);
		orderedRects.push_back(r);
	}

	sort(orderedRects.begin(), orderedRects.end(), sort_y);
	for (size_t i = 0; i < 9; i++){
		int offset = i*9;
		vector<Rect>::iterator start = orderedRects.begin()+offset;
		vector<Rect>::iterator end = orderedRects.begin()+offset+9;
		sort(start, end, sort_x);
	}

}

//Converts to binary black and white image
void process_img(const Mat& in, Mat& out){
	Mat tmp_img, tmp_img2;
	cvtColor(in, tmp_img, CV_BGR2GRAY);
	GaussianBlur(tmp_img, tmp_img2, Size(3, 3), 0, 0);
	adaptiveThreshold(tmp_img2, out, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
}

//Used to scale the character images to a standard size to be unsed i nthe knearest model
//this adds whitespace to the image to avoid distorting the dimentions
Mat scaleAddWhitespace(Mat img, int target_width, int target_height){
	float width = img.cols;
	float height = img.rows;

	Mat target_img = Mat::zeros(target_width, target_height, img.type());
	float x_scale = width/target_width;
	float y_scale = height/target_height;

	Mat scaled;
	Size s;
	if (x_scale > 1 && y_scale < 1){
		//Scale x to target
		s.width = target_width;
		s.height = height/x_scale;
	}
	else if (x_scale < 1 && y_scale > 1) {
		//Scale y to target
		s.width = width/y_scale;
		s.height = target_height;
	}
	else if ((x_scale-1) > (y_scale-1)) {
		//width is closer, scale y to target
		s.width = target_width;
		s.height = height/x_scale;
	}
	else {
		s.width = width/y_scale;
		s.height = target_height;
	}

	resize(img, target_img, s);

	Mat larger(Size(target_width,target_height),target_img.type());
	larger = Scalar(255);
	target_img.copyTo(larger(Rect(0,0,target_img.cols,target_img.rows)));

	Mat t, blur;
	GaussianBlur(larger,blur,Size(5,5), 0, 0);
	adaptiveThreshold(blur, t, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 1);

	Mat ret_img;

	threshold(larger, ret_img, 150, 255, THRESH_BINARY);

	return ret_img;
}

Mat toSingleLine(const Mat& image){
	Mat tmp, tmp2, ret_img;
	tmp = image.clone();
	tmp.convertTo(tmp2, CV_32FC1);
	ret_img = tmp2.reshape(0,1);
	return ret_img;
}

int main(int argc, char** argv){

	bool learning = false;

	if (argc < 2 || argc > 3){
		cout<<"Incorrect number of arguments"<<endl;
		exit(-1);
	}

	Mat image = imread(argv[1], CV_32FC1);

	//If we cant read the image, exit
	if (!image.data){
		cout<<"Problem reading image"<<endl;
		return -1;
	}

	//If we are learning, set iup a window
	if (argc == 3 && strcmp(argv[2], "learn") == 0){
		learning = true;
		namedWindow("Learn Image", CV_WINDOW_AUTOSIZE);
	}

	//Figure out how many images we have to learn from
	setFileCounts();
	int learnCnt = 0;
	for (int i=0; i<=9; i++){
		learnCnt += fileCounts[i];
	}

	//If we are not learning, set up the knearest model
	KNearest kn_model;
	if (!learning && learnCnt > 0){
		Mat _samples(learnCnt,sample_area,CV_32FC1);
		Mat _responses(learnCnt, 1, CV_32FC1);

		char infile[256];
		for (unsigned i=1; i<=9; i++){
			unsigned j = 0;
			while (true){
				Mat gray;
				sprintf(infile, "./learn/%d_%d.jpg", i, j);
				Mat read_img = imread(infile, CV_32FC1);
				if (!read_img.data){
					break;
				}
				cvtColor(read_img, gray, CV_BGR2GRAY);
				Mat learn_img = toSingleLine(gray);

				_samples.push_back(learn_img);
				_responses.push_back(Mat(1, 1, CV_32FC1, i));
				j++;
			}
		}
		kn_model.train(_samples, _responses);
	}
    
	//Parse the image for the 81 cells of a sudoku game
	//Try different theshold values until 81 squares are found
	vector<vector<Point> > squares;
	int thresh_val = 50;
	while (true){
		squares.clear();
		findSquares(image, squares, thresh_val);
		if (squares.size() == 81){
			break;
		}
		thresh_val+=1;
		if (thresh_val > 255){
			cout<<"Error parsing image, couldn't find cells"<<endl;
			return -1;
		}
	}

	//Convert the squares into an ordered vector of rects
	vector<Rect> orderedRects; 
	sortConvertSquares(squares, orderedRects);

	//Initialize the game to all zeros
	int game[9][9];
	for (unsigned i=0; i<9; i++){
		for (unsigned j=0; j<9; j++){
			game[i][j] = 0;
		}
	}

	//Process each cell
	for ( int i=0; i < orderedRects.size(); i++){

		Mat cropped = image(orderedRects[i]);
		Mat processed,bordered;
		process_img(cropped, processed);

		//Add a small border of whitespace around the cell to help with analysis 
		copyMakeBorder(processed, bordered, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(255));

		//Find the contours of the cell, and sort them by size of their bounding boxes, descending
		Mat bordered_copy = bordered.clone();
		vector<vector<Point> > cell_contours;
		findContours(bordered, cell_contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
		sort(cell_contours.begin(), cell_contours.end(), sort_bound_area);

		//Attempt to find the contour of the character
 		for (int j=0; j<cell_contours.size(); j++){
			Point2f center;
			float radius;
			minEnclosingCircle((Mat)cell_contours[j], center, radius);
			radius *= .9;

			//Make sure that the min enclosing circle of the contour is contained within the cell
			//this helps filter out noise on the sides of the cell
			if (center.x+radius < bordered.cols &&
				center.x-radius > 0 &&
				center.y+radius < bordered.rows &&
				center.y-radius > 0 &&
				//Also make sure the center of the cell is contained within the enclosing circle
				bordered.cols/2 > center.x - radius &&
				bordered.cols/2 < center.x + radius &&
				bordered.rows/2 > center.y - radius &&
				bordered.rows/2 < center.y + radius) {

				//Crop out the character, and process it
				Rect roi = boundingRect(Mat(cell_contours[j]));
				Mat bound_number = bordered_copy(roi);
				Mat scaled = scaleAddWhitespace(bound_number,sample_width,sample_height);

				if (learning){
					//If in learning mode, present the image and save the result
					imshow("Learn Image", scaled);
					char c = waitKey();
					save_learned(scaled, c);
				}
				else {
					//If not in learning mode, figure out the character
					Mat learn_img = toSingleLine(scaled);
					float val = kn_model.find_nearest(learn_img, 1);
					game[i/9][i%9] = val;
				}
				break;
			}
		}
	}

	if (!learning){

		//Made a copy of the game
		int game_copy[9][9];
		for (unsigned i=0; i<9; i++){
			for (unsigned j=0; j<9; j++){
				game_copy[i][j] = game[i][j];
			}
		}

		//Run the logic to solve the game
		bool success = solveGame(game);
		
		if (success){
			//If success, print the results on the game
			for (int i=0; i<81; i++){
				int x = i/9;
				int y = i%9;
				Rect r = orderedRects[i];

				if (game_copy[x][y] == 0){
					ostringstream os;
					os<<game[x][y];
					int baseline=0;
					Size s = getTextSize(os.str(), FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
					float xscale = (float)r.width/(float)s.width;
					float yscale = (float)r.height/(float)s.height;
					float scale = min(xscale, yscale)*.75;

					int move_x = (float)r.width/2 - (((float)s.width*scale)/2);
					int move_y = (float)r.height/2 - (((float)s.height*scale)/2);

					putText(image, os.str(), Point(r.x+move_x, r.y+r.height-move_y), FONT_HERSHEY_SIMPLEX, scale, Scalar(255,0,0), 2);
				}
			}

			ostringstream os;
			os<<"solved_"<<argv[1];
			imwrite(os.str(), image);
			cout<<"Success, solved game saved to disk"<<endl;
		}
		else{
			cout<<"Printing digit recognition for debug"<<endl;
			cout<<"If output doesn't match input image, run learn command on current image"<<endl;
			for (unsigned i=0; i<9; i++){
				for (unsigned j=0; j<9; j++){
					if (game[i][j] == 0){
						cout<<".";
					}
					else{
						cout<<game[i][j];
					}
				}
				cout<<endl;
			}
		}
	}
    return 0;
}