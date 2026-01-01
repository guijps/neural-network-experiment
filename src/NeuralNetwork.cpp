#include "../include/NeuralNetwork.hpp"
#include "../include/utils/MultiplyMatrix.hpp"
int INPUT_LAYER_INDEX = 0;
NeuralNetwork::NeuralNetwork(vector<int> topology)
{
    this->topologySize = topology.size();
    this->topology = topology;
    for(int i = 0; i< topologySize; i++){
        Layer *layer = new Layer(topology[i]);
        this->layers.push_back(layer);
    }

    for(int i =0; i < topologySize -1; i++){
        Matrix *weightMatrix = new Matrix(topology[i],topology[i+1], true);
        this->weightMatrices.push_back(weightMatrix);
    }
    
}

void NeuralNetwork::printWeightsToConsole()
{
    for(int i = 0; i < this->weightMatrices.size(); i++){
        this->weightMatrices[i]->printToConsole();
    }
    cout << endl;
}

void NeuralNetwork::setCurrentInput(vector<double> inputValues)
{
    cout << "Setting current input values." << endl;
    this->currentInput = inputValues;
    for(int i = 0; i< topology[INPUT_LAYER_INDEX]; i++){
        this->layers[INPUT_LAYER_INDEX]->setValue(i,inputValues[i]);
    }
}

void NeuralNetwork::setCurrentTarget(vector<double> targetValues)
{
    cout << "Setting current target values." << endl;
    this->target = targetValues;
       if(errors.size() >0)
       {
            fill(errors.begin(), errors.end(), 0.0);

            cout << "Finished setting current target values." << endl;
            return;
       }
    //fill errors vector with zeros
    for(int i =0; i < topology.back(); i++)
    {
        this->errors.push_back(0.0);
    }
    cout << "Finished setting current target values." << endl;
}


void NeuralNetwork::printToConsole()
{
    cout << "Neural Network Topology: ";
    for(int i = 0; i < this->layers.size(); i++){
        cout<<"Layer " << i << endl;
        if(i==0)
        {
            Matrix *m = this->layers[i]->matrixifyVals();
            m->printToConsole();
        }
        else
        {
            Matrix *m = this->layers[i]->matrixifyActivatedVals();
            m->printToConsole();
        }


    }
    cout << endl;
}

void NeuralNetwork::feedForward()
{
    int n = this-> layers.size();
    for(int i = 0; i< ( n-1); i++)
    {
        // cout<<"------------------------"<<endl;
        // cout<<"Feedforwarding from Layer "<< i <<endl;
        Matrix* a = this->getNeuronMatrix(i);
        
        if(i != 0)
        {
            a = this-> getActivatedNeuronMatrix(i); 
        }
        // cout<<"Neuron Matrix: "<< i<< endl;
        // a->printToConsole();
        Matrix *b = this-> getWeightMatrix(i);
        // cout<<"Weight Matrix: "<< i<< endl;
        // b->printToConsole();
        Matrix *c = (new utils::MultiplyMatrix(a,b))->execute();
        
        for(int c_index= 0;c_index< c->getCols();c_index++){
            this->setNeuronValue(i+1, c_index, c->getValue(0,c_index));
        }
    }
}

void NeuralNetwork::setErrors()
{
    //assert that the NN already have targets
    if(this->target.size() == 0){
        cerr << "Error: No target values set for the neural network." << endl;
        return;
    }

    if(this->target.size() != this->layers.at(this->topologySize -1)->getNeurons().size()){
        cerr << "Error: Target size does not match output layer size." << endl;
        return;
    }

    this->error = 0.0;
    int outputLayerIndex = this->topologySize -1;
    vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();
    for(int i = 0; i <target.size(); i++){
       
        // Minimize Error = Target 
        double neuronOutput = outputNeurons.at(i)->getActivatedValue();
        double targetValue = this->target.at(i);
        double neuronError = targetValue - neuronOutput;
        this->errors.at(i) = neuronError*neuronError; //Using Mean Squared Error
        this->error +=  neuronError ; //Using Mean Squared Error
    }
    //this is for historical tracking
    this->historicalErrors.push_back(this->error);
}

void NeuralNetwork::backPropagate()
{
    setErrors();
    vector<Matrix*> newWeights;
    Matrix* gradients;
    //output to hidden layers
    //get derived values from given Layers, and we can get the output layer from outputlayer values.
    int outputlayerIndex = this->layers.size() -1;
    Matrix* derivedValuesYtoZ = this->layers.at(outputlayerIndex)->matrixifyDerivedVals();
    Matrix* gradientsYtoZ = new Matrix(1, this->layers.at(outputlayerIndex)->getNeurons().size(), false);
    for(int i= 0 ; i < this->errors.size(); i++)
    {
        double derivedValue = derivedValuesYtoZ->getValue(0,i);
        double errorValue = this->errors.at(i);

        double gradient = derivedValue * errorValue;
        gradientsYtoZ->setValue(0,i, gradient);
    }

    int lastHiddenLayerIndex = outputlayerIndex -1;
    Layer *lastHiddenLayer = this->layers.at(lastHiddenLayerIndex);
    Matrix* weightsOutputToHidden = this->weightMatrices.at(lastHiddenLayerIndex);
    Matrix* deltaOutputHidden = (new utils::MultiplyMatrix(gradientsYtoZ->transpose(),lastHiddenLayer->matrixifyActivatedVals()))->execute()->transpose();
    Matrix* newWeightsOutputToHidden = new Matrix(deltaOutputHidden->getRows(), deltaOutputHidden->getCols(), false);
    
    for(int r =0; r < deltaOutputHidden->getRows(); r++)
    {
        for(int c=0; c < deltaOutputHidden->getCols(); c++)
        {
            double oldWeight    = weightsOutputToHidden->getValue(r,c);
            double deltaWeight  = deltaOutputHidden->getValue(r,c);
            double newWeight    = oldWeight - deltaWeight; //Learning Rate is assumed to be 1 for simplicity
            newWeightsOutputToHidden->setValue(r,c,newWeight);
        }
    }
    newWeights.push_back(newWeightsOutputToHidden);
    
    gradients= new Matrix(gradientsYtoZ->getRows(), gradientsYtoZ->getCols(), false);
    for(int i =0; i < gradientsYtoZ->getRows(); i++)
    {
        for(int j =0; j < gradientsYtoZ->getCols(); j++)
        {
            gradients->setValue(i,j, gradientsYtoZ->getValue(i,j));
        }
    }

    for(int i= lastHiddenLayerIndex; i>0;i--)
    {
        Layer* l =this->layers.at(i);
        Matrix* derivedHidden = l->matrixifyDerivedVals();
        Matrix* activatedHidden = l->matrixifyActivatedVals();
        Matrix* derivedGradients = new Matrix(1, l->getNeurons().size(), false);

        Matrix* weightsMatrix = this->weightMatrices.at(i);
        Matrix* originalWeight = this->weightMatrices.at(i-1);
        
        for(int r= 0; r< weightsMatrix->getRows(); r++)
        {
            double sum =0.0;
            for(int c=0; c < weightsMatrix->getCols(); c++)
            {
                double w = weightsMatrix->getValue(r,c);
                double g = gradients->getValue(0,c);
                sum += w * g;
            }
            
            double activatedVal = activatedHidden->getValue(0,r);
            double gradient = activatedVal * sum;
            derivedGradients->setValue(0,r, gradient);
        }
        
        Matrix* leftNeurons = (i-1)==0 ? this->layers.at(0)->matrixifyVals() : this->layers.at(i-1)->matrixifyActivatedVals();
        Matrix* deltaWeights = (new utils::MultiplyMatrix(derivedGradients->transpose(), leftNeurons))->execute()->transpose();
        Matrix * newWeightsHidden = new Matrix(deltaWeights->getRows(), deltaWeights->getCols(), false);
        for(int r =0; r < newWeightsHidden->getRows(); r++)
        {
            for(int c=0; c < newWeightsHidden->getCols(); c++)
            {
                double oldWeight    = originalWeight->getValue(r,c);
                double deltaWeight  = deltaWeights->getValue(r,c);
                double newWeight    = oldWeight - deltaWeight; //Learning Rate is assumed to be 1 for simplicity
                newWeightsHidden->setValue(r,c,newWeight);
            }
        } 
        newWeights.push_back(newWeightsHidden);

        gradients= new Matrix(derivedGradients->getRows(), derivedGradients->getCols(), false);
        for(int i =0; i < derivedGradients->getRows(); i++)
        {
            for(int j =0; j < derivedGradients->getCols(); j++)
            {
                gradients->setValue(i,j, derivedGradients->getValue(i,j));
            }
        }
        
    }
    cout<<"Completed backpropagation." << endl;
    cout<<"New Weights Size: "<< newWeights.size() << endl;
    cout<<"Old Weights Size: "<< this->weightMatrices.size() << endl;
    
    reverse(newWeights.begin(), newWeights.end());

    this->weightMatrices = newWeights;
}