#include <iostream>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/utils/MultiplyMatrix.hpp"
#include <assert.h>
using namespace std;

#define EPOCHS 10

bool test_feed_forward()
{
    
    vector<double> input;
    input.push_back(1);
    input.push_back(0);
    input.push_back(1);

    /*Input
    [
        1 0 1
    ]
    */
    // Weight Matrix topology is 3,2,3 so the first weight matrix will be 3,2 and the cedond 2,3
    // because we will have a 1x3 * 3x2 => 1x2 and then 1x2*2x3 => 1x3 the output.
    //Weights Layer 0to1
    /*
        0.5   0.5
        0.5   0.5
        0.5   0.5
    */
    //Weights Layer 1to2
    /*
        0.5   0.5   0.5
        0.5   0.5   0.5
    */

    vector<int> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(3);

    NeuralNetwork nn(topology);
    nn.setCurrentInput(input);
    Matrix* weights = nn.getWeightMatrix(0);
    weights->setValue(0,0,0.5);
    weights->setValue(0,1,0.5);
    weights->setValue(1,0,0.5);
    weights->setValue(1,1,0.5);
    weights->setValue(2,0,0.5);
    weights->setValue(2,1,0.5);

    Matrix* weights2 = nn.getWeightMatrix(1);
    weights2->setValue(0,0,0.5);
    weights2->setValue(0,1,0.5);
    weights2->setValue(0,2,0.5);
    weights2->setValue(1,0,0.5);
    weights2->setValue(1,1,0.5);
    weights2->setValue(1,2,0.5);
    nn.feedForward();
    Matrix* hiddenLayer = nn.getActivatedNeuronMatrix(1);
    Matrix* outputLayer = nn.getActivatedNeuronMatrix(2);
    //Expected Hidden Layer after feedforward
    assert(abs(hiddenLayer->getValue(0,0)-0.5) < 0.0001); //  (1*0.5 + 0*0.5 +1*0.5) =1.0 => 1/(1+|1|)=0.5
    assert(abs(hiddenLayer->getValue(0,1)-0.5) < 0.0001); //  (1*0.5 + 0*0.5 +1*0.5) =1.0 => 1/(1+|1|)=0.5
    
    assert(abs(outputLayer->getValue(0,0)-0.3333) < 0.0001); //  (1*0.5 + 0*0.5 +1*0.5) =1.0 => 1/(1+|1|)=0.5
    assert(abs(outputLayer->getValue(0,1)-0.3333) < 0.0001); //  (1*0.5 + 0*0.5 +1*0.5) =1.0 => 1/(1+|1|)=0.5
    assert(abs(outputLayer->getValue(0,2)-0.3333) < 0.0001); //  (1*0.5 + 0*0.5 +1*0.5) =1.0 => 1/(1+|1|)=0.5
    
    return true;

}

bool test_matrix_mult_nxn(){
    Matrix* m1 = new Matrix(3,3,true);
    m1->setValue(0,0,1);
    m1->setValue(0,1,2);
    m1->setValue(0,2,3);
    m1->setValue(1,0,4);
    m1->setValue(1,1,5);
    m1->setValue(1,2,6);
    m1->setValue(2,0,7);
    m1->setValue(2,1,8);
    m1->setValue(2,2,9);
    Matrix* m2 = new Matrix(3,3,true);
    m2->setValue(0,0,1);
    m2->setValue(0,1,2);
    m2->setValue(0,2,3);
    m2->setValue(1,0,4);
    m2->setValue(1,1,5);
    m2->setValue(1,2,6);
    m2->setValue(2,0,7);
    m2->setValue(2,1,8);
    m2->setValue(2,2,9);
    utils::MultiplyMatrix mm(m1,m2);
    Matrix* result = mm.execute();
    if(result->getRows() !=3 || result->getCols() !=3){
        cout << "Matrix multiplication result has incorrect dimensions." << endl;
        return false;
    }
    //
    assert(result->getValue(0,0) == 30); // 1*1 +2*4 +3*7 =30
    assert(result->getValue(0,1) == 36); // 1*2 +2*5 +3*8 =36
    assert(result->getValue(0,2) == 42); // 1*3 +2*6 +3*9 =42   
    assert(result->getValue(1,0) == 66); // 4*1 +5*4 +6*7 =66
    assert(result->getValue(1,1) == 81); // 4*2 +5*5 +6*8 =81
    assert(result->getValue(1,2) == 96); // 4*3 +5*6 +6*9 =96
    assert(result->getValue(2,0) == 102); // 7*1 +8*4 +9*7 =102
    assert(result->getValue(2,1) == 126); // 7*2 +8*5 +9*8 =126
    assert(result->getValue(2,2) == 150); // 7*3 +8*6 +9*9 =150
    return true;
}

bool test_matrix_mult_inequal_1(){
    Matrix m1(3,2,true);
    m1.setValue(0,0,1);
    m1.setValue(0,1,2);
    m1.setValue(1,0,3);
    m1.setValue(1,1,4);
    m1.setValue(2,0,5);
    m1.setValue(2,1,6);
    Matrix m2(2,3,true);
    m2.setValue(0,0,1);
    m2.setValue(0,1,2);
    m2.setValue(0,2,3);
    m2.setValue(1,0,4);
    m2.setValue(1,1,5);
    m2.setValue(1,2,6);
    utils::MultiplyMatrix mm(&m1,&m2);
    Matrix* result = mm.execute();
    if(result->getRows() !=2 || result->getCols() !=2){
        return false;
    }
    assert(result->getValue(0,0) == 9); // 1*1 +2*4 =9
    assert(result->getValue(0,1) == 12); // 1*2 +2*5 =12
    assert(result->getValue(0,2) == 15); // 1*3 +2*6 =15   
    assert(result->getValue(1,0) == 19); // 3*1 +4*4 =19
    assert(result->getValue(1,1) == 26); // 3*2 +4*5 =26
    assert(result->getValue(1,2) == 33); // 3*3 +4*6 =33
    assert(result->getValue(2,0) == 29); // 5*1 +6*4 =29
    assert(result->getValue(2,1) == 40); // 5*2 +6*5 =40
    assert(result->getValue(2,2) == 51); // 5*3 +6*6 =51
    return true;
}

bool test_matrix_mult_inequal_2(){

    Matrix m1(2,3,true);
    m1.setValue(0,0,1);
    m1.setValue(0,1,2);
    m1.setValue(0,2,3);
    m1.setValue(1,0,4);
    m1.setValue(1,1,5);
    m1.setValue(1,2,6);
    Matrix m2(3,2,true);
    m2.setValue(0,0,1);
    m2.setValue(0,1,2);
    m2.setValue(1,0,3);
    m2.setValue(1,1,4);
    m2.setValue(2,0,5);
    m2.setValue(2,1,6);
    utils::MultiplyMatrix mm(&m1,&m2);
    Matrix* result = mm.execute();
    if(result->getRows() !=2 || result->getCols() !=2){
        return false;
    }

    assert(result->getValue(0,0) == 22); // 1*1 +2*3 +3*5 =22
    assert(result->getValue(0,1) == 28); // 1*2 +2*4 +3*6 =28
    assert(result->getValue(1,0) == 49); // 4*1 +5*3 +6*5 =49
    assert(result->getValue(1,1) == 64); // 4*2 +5*4 +6*6 =64
    return true;
}

bool test_matrix(){
    cout << "Starting Matrix tests..." << endl;
    test_matrix_mult_nxn();
    cout << "Completed test_matrix_mult_nxn." << endl;
    test_matrix_mult_inequal_1();
    cout << "Completed test_matrix_mult_inequal_1." << endl;
    test_matrix_mult_inequal_2();
    cout << "Completed test_matrix_mult_inequal_2." << endl;
    return true;
}

bool test_back_propagation()
{
    //to be done
    return true;
}

bool xor_test()
{

    vector<double> input;
    input.push_back(0);
    input.push_back(0);

    vector<double> output;
    output.push_back(0);

    /*Input
    [
        1 0 1
    ]
    */
    // Weight Matrix topology is 3,2,3 so the first weight matrix will be 3,2 and the cedond 2,3
    // because we will have a 1x3 * 3x2 => 1x2 and then 1x2*2x3 => 1x3 the output.
    //Weights Layer 0to1
    /*
        0.5   0.5
        0.5   0.5
        0.5   0.5
    */
    //Weights Layer 1to2
    /*
        0.5   0.5   0.5
        0.5   0.5   0.5
    */

    vector<int> topology;
    topology.push_back(2);
    topology.push_back(2);
    topology.push_back(1);

    NeuralNetwork nn(topology);
    for(int i =0;i< EPOCHS;i++)
    {
    
        input[0]=0;
        input[1]=0;
        output[0]=0;
        nn.setCurrentInput(input);
        nn.setCurrentTarget(output);
        //Trainning process
        nn.Train(EPOCHS);
        cout<< "Training 1 Completed." << endl;
        //--- change Input to 01->1
        input[0]=0;
        input[1]=1;
        output[0]=1;

        nn.setCurrentInput(input);
        nn.setCurrentTarget(output);
        
        nn.Train(EPOCHS);
        cout<< "Training 2 Completed." << endl;
        //--- change Input to 11->0
        input[0]=1;
        input[1]=1;
        output[0]=0;

        nn.setCurrentInput(input);
        nn.setCurrentTarget(output);

        nn.Train(EPOCHS);
        cout<< "Training 3 Completed." << endl;
        //--- change Input to 10->1
        input[0]=1;
        input[1]=0;
        output[0]=1;

        nn.setCurrentInput(input);
        nn.setCurrentTarget(output);

        nn.Train(EPOCHS);
    }


    input[0]=0;
    input[1]=0;
    nn.setCurrentInput(input);
    nn.feedForward();
    Matrix* outputLayer1 = nn.getActivatedNeuronMatrix(2);
    cout << "Output for 00:" << endl;
    outputLayer1->printToConsole();
    double a = outputLayer1->getValue(0,0);
    //------------------------
    input[0]=1;
    input[1]=0;
    nn.setCurrentInput(input);
    nn.feedForward();
    Matrix* outputLayer2 = nn.getActivatedNeuronMatrix(2);
    cout << "Output for 01:" << endl;
    outputLayer2->printToConsole();
    double b = outputLayer2->getValue(0,0);
    //------------------------
    input[0]=1;
    input[1]=1;
    nn.setCurrentInput(input);
    nn.feedForward();
    Matrix* outputLayer3 = nn.getActivatedNeuronMatrix(2);
    cout << "Output for 11:" << endl;
    outputLayer3->printToConsole();
    double c = outputLayer3->getValue(0,0);
    //------------------------
    input[0]=0;
    input[1]=1;
    nn.setCurrentInput(input);
    nn.feedForward();
    Matrix* outputLayer4 = nn.getActivatedNeuronMatrix(2);
    cout << "Output for 10:" << endl;
    outputLayer4->printToConsole();
    double d = outputLayer4->getValue(0,0);

    return true;
}

int main(int argc, char** argv)
{
    
    if(!test_matrix())
        cout << "Matrix test failed." << endl;
        
    // if(!test_feed_forward())
    //     cout << "Feedforward test failed." << endl;
        
    // if(!test_back_propagation())
    //         cout << "Backpropagation test failed." << endl;
    
    // if(!xor_test())
    //     cout << "XOR test failed." << endl;
        
    cout<< "All tests completed." << endl;
    return 0;
 }
