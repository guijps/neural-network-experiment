#include "../include/Layer.hpp"


Layer::Layer(int size){
    this->size = size;
    for(int i = 0; i < size; i++){
        Neuron *neuron = new Neuron(0.0);
        this->neurons.push_back(neuron);
    }
} 

void Layer::setValue(int index, double val){
    this->neurons.at(index)->setValue(val);
}

Matrix* Layer::matrixifyVals(){
    Matrix* m = new Matrix(1,neurons.size(),false);
    for(int i = 0; i < neurons.size(); i++){
        m->setValue(0,i,this->neurons.at(i)->getValue());
    }
    return m;
}

Matrix* Layer::matrixifyActivatedVals(){
    Matrix* m = new Matrix(1,neurons.size(),false);
    for(int i = 0; i < neurons.size(); i++){
        m->setValue(0,i,this->neurons.at(i)->getActivatedValue());
    }
    return m;
}

Matrix* Layer::matrixifyDerivedVals(){
    Matrix* m = new Matrix(1,neurons.size(),false);
    for(int i = 0; i < neurons.size(); i++){
        m->setValue(0,i,this->neurons.at(i)->getDerivedValue());
    }
    return m;
}