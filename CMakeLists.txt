project(neural-network-experiment)
set(CMAKE_BUILD_TYPE Debug)

add_executable(
    neural-network-experiment
    src/main.cpp
    src/Neuron.cpp
    src/Matrix.cpp
    src/NeuralNetwork.cpp
    src/Layer.cpp
    src/utils/MultiplyMatrix.cpp
)

add_executable(
    test
    src/test_nn.cpp
    src/Neuron.cpp
    src/Matrix.cpp
    src/NeuralNetwork.cpp
    src/Layer.cpp
    src/utils/MultiplyMatrix.cpp
)