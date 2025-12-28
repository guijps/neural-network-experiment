#ifndef _NEURON_HPP_
#define _NEURON_HPP_
#include <iostream>
using namespace std;

class Neuron 
{
    public:
        Neuron(double val);

        void setValue(double val);
        //Fast Sigmoid Function -Easyest function- Non Linear Function
        //f(x) = x / (1 + abs(x))
        void activate();

        //Derivative of Fast Sigmoid Function
        //f'(x) = f(x) * (1 - f(x))
        void derive();

        double getValue() {return this->value;}
        double getActivatedValue() {return this->activatedValue;}
        double getDerivedValue() {return this->derivedValue;}
    //Given a Neurons a raw value, we need 
    //an activation funcion that will fit it
    //into a fixed range value
    private:
        //1.5
        double value;

        //0-1 - from activation function
        double activatedValue;

        //will get derivative of activation function
        double derivedValue;

};

#endif //_NEURON_HPP_