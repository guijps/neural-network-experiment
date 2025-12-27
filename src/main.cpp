#include <iostream>
#include "../include/Neuron.hpp"

using namespace std;

int main(int argc, char** argv)
{
    cout << "Neural Network Experiment Main" << endl;
    Neuron n(1.5);
    cout << "Neuron Raw Value: " << n.getValue() << endl;
    cout << "Neuron Activated Value: " << n.getActivatedValue() << endl;
    cout << "Neuron Derived Value: " << n.getDerivedValue() << endl;

    return 0;
}