#include <iostream>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/MultiplyMatrix.hpp"
using namespace std;

int main(int argc, char** argv)
{
    //-------------------------
    // cout << "Neural Network Experiment Main" << endl;
    // Neuron n(1.5);
    // cout << "Neuron Raw Value: " << n.getValue() << endl;
    // cout << "Neuron Activated Value: " << n.getActivatedValue() << endl;
    // cout << "Neuron Derived Value: " << n.getDerivedValue() << endl;
    // Matrix m(3,2,true);
    // m.printToConsole();
    // Matrix* mt = m.transpose();
    // cout << "Transposed Matrix:" << endl;
    // mt->printToConsole();

    //----------------------
    vector<double> input;
    input.push_back(1);
    input.push_back(0);
    input.push_back(1);

    vector<int> topology;
    topology.push_back(3); 
    topology.push_back(2);
    topology.push_back(3);

    NeuralNetwork*  nn = new NeuralNetwork(topology);
    nn->setCurrentInput(input);
    nn->setCurrentTarget(input);

    //Trainning process
    for(int i =0; i < 100; i++){
        cout << "----- Epock " << i+1 << " -----" << endl;
        nn->feedForward();
        nn->setErrors();
        nn->backPropagate();
        cout << "Total Error: " << nn->getTotalError() << endl;
    }
    //----------------------

    // Matrix* m1 = new Matrix(2,3,true);
    // Matrix* m2 = new Matrix(3,2,true);
    // cout << "Matrix 1:" << endl;
    // m1->printToConsole();
    // cout << "Matrix 2:" << endl;
    // m2->printToConsole();
    // utils::MultiplyMatrix mm(m1,m2);
    // Matrix* result = mm.execute();
    // cout << "Resultant Matrix after Multiplication:" << endl;
    // result->printToConsole();
    return 0;
}