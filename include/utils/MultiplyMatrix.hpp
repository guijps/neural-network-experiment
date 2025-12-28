#ifndef _MULTIPLY_MATRIX_HPP_
#define _MULTIPLY_MATRIX_HPP_

#include "../Matrix.hpp"
using namespace std;
namespace utils 
{
    class MultiplyMatrix
    {
        public:
            MultiplyMatrix(Matrix* m1, Matrix* m2);
            Matrix *execute();
        private:
            Matrix* m1;
            Matrix* m2;
            Matrix* result;
    };
}

#endif //_MULTIPLY_MATRIX_HPP_