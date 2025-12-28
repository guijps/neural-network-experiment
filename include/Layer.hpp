#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include <vector>
#include "Neuron.hpp"
#include "Matrix.hpp"
using namespace std;

class Layer{
    public:
        Layer(int size);
        void setValue(int index, double val);
        Matrix *  matrixifyVals(); //to be implemented
        Matrix *  matrixifyActivatedVals(); //to be implemented
        Matrix *  matrixifyDerivedVals(); //to be implemented
        vector<Neuron *> getNeurons(){ return this->neurons; }
    private:
        int size;
        vector<Neuron *> neurons;

};
#endif //_LAYER_HPP_