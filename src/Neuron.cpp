#include "../include/Neuron.hpp"
//Constructor
Neuron::Neuron(double val)
{
    //given a valuem stroe it, activate it and derive it
    this->value = val;
    activate();
    derive();
}

void Neuron::activate()
{
    //Fast Sigmoid Function -Easyest function- Non Linear Function
    //f(x) = x / (1 + abs(x))
    this->activatedValue = this->value / (1 + abs(this->value));
}

void Neuron::derive()
{
    //Derivative of Fast Sigmoid Function
    //f'(x) = f(x) * (1 - f(x))
    this->derivedValue = this->activatedValue * (1 - this->activatedValue);
}