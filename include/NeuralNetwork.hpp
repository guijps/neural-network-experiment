#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include "Matrix.hpp"
#include "Neuron.hpp"
#include <vector>
#include "Layer.hpp"
#include <algorithm>
using namespace std;

class NeuralNetwork
{
    public:
        NeuralNetwork(vector<int> topology);
        void setCurrentInput(vector<double> inputValues);
        void printToConsole();
        void printWeightsToConsole();
        void feedForward();
        void backPropagate();
        void setCurrentTarget(vector<double> targetValues);
        void setErrors();

        void Train(int epochs)
        {
            for(int i =0; i < epochs; i++){
                this->feedForward();
                this->setErrors();
                this->backPropagate();
            }
        }
        void setNeuronValue(int indexLayer,int indexNeuron,double value)
        {
            this->layers.at(indexLayer)->setValue(indexNeuron,value);
        }
        Matrix* getNeuronMatrix(int layerIndex){
            return this->layers.at(layerIndex)->matrixifyVals();
        }
        Matrix* getActivatedNeuronMatrix(int layerIndex){
            return this->layers.at(layerIndex)->matrixifyActivatedVals();
        }
        Matrix* getWeightMatrix(int index){
            return this->weightMatrices.at(index);
        }   
        double getTotalError(){return this->error;}
        vector<double> getErrors(){return this->errors;}
    private:
        int topologySize;
        vector<int>         topology;
        vector<Layer *>     layers;
        vector<Matrix *>    weightMatrices;
        vector<Matrix *>    gradientMatrices;
        vector<double>      currentInput;
        vector<double>      target;
        double              error;
        vector<double>      errors;
        vector<double>      historicalErrors; 
};
#endif //_NEURAL_NETWORK_HPP_