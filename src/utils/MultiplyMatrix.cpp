#include "../../include/utils/MultiplyMatrix.hpp"
utils::MultiplyMatrix::MultiplyMatrix(Matrix* m1, Matrix* m2)
{
    if( m1->getCols() != m2->getRows() )
    {
        throw invalid_argument("Incompatible matrix dimensions for multiplication.");
    }

    this->m1 = m1;
    this->m2 = m2;
    result = new Matrix(m1->getRows(), m2->getCols(), false);
}

Matrix* utils::MultiplyMatrix::execute()
{
    int resultRows = m1->getRows();
    int resultCols = m2->getCols();

    for(int i = 0; i < resultRows; i++)
    {
        for(int j = 0; j < resultCols; j++)
        {
            double sum = 0.0;
            for(int k = 0; k < m1->getCols(); k++)
            {
                sum += m1->getValue(i, k) * m2->getValue(k, j);
            }
            result->setValue(i, j, sum);
        }
    }

    return result;
}