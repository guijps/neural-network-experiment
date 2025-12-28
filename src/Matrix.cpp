#include "../include/Matrix.hpp"
#include <random>

Matrix::Matrix(int rows,int cols, bool isRandom)
{
    this->rows = rows;
    this->cols = cols;
    for(int i = 0; i < rows; i++){
        vector<double> rowValues;
        for(int j = 0; j < cols; j++){
            double r = 0.00;
            if(isRandom){

                r = generateRandomNumber();
            }
            
            rowValues.push_back(r);
        
        }
        values.push_back(rowValues);
    }
}

Matrix * Matrix::transpose(){
    Matrix* transposed = new Matrix(cols, rows, false);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            transposed->setValue(j, i, this->getValue(i, j));
        }
    }
    return transposed;
}

void Matrix::printToConsole()
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            cout << values[i][j] << "\t\t";
        }
        cout << endl;
    }
}

void Matrix::setValue(int row, int col, double val)
{
    values.at(row).at(col) = val;
}

double Matrix::getValue(int row, int col)
{
    return values.at(row).at(col);
}
double Matrix::generateRandomNumber()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
}