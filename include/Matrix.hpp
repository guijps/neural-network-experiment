#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <iostream>
#include <vector>
using namespace std;
class Matrix
{
    public:
        Matrix(int rows,int cols, bool isRandom);
        Matrix * transpose();
        void setValue(int row, int col, double val);
        double getValue(int row, int col);
        double generateRandomNumber();
        void printToConsole();
        int getRows() { return rows; }
        int getCols() { return cols; }

    private:
        int rows;
        int cols;
        vector<vector<double>> values;
};
#endif //_MATRIX_HPP_