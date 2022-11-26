# qcnn-data-classification
# Quantum Convolution neural networks : data classification

This is an implementation of 8 qubit Quantum convolutional neural network for classical data classification. 
It uses Pennylane software (https://pennylane.ai) for classifying EMNIST, MNIST and Fashion MNIST datasets.

Prerequisites:
1. Install Anaconda for python
2. setup spyder or PyCharm or any other Python development framework
3. install scikit-learn scipy matplotlib scikit-learn-intelex emnist packages

Since the programme is using pennylane, please go through the pennylane recommendations for how to run it on GPU or multi CPU

### 1. Data Preparation
**"data.py"**: loads dataset (EMNIST : letters, digits, mnist, balanced, byclass, bymerge, MNIST or Fashion MNIST). It is also doing cross dataset training and testing for MNIST datasets) after classical preprocessing. 
"feature_reduction" argument contains information of preprocessing method and     dimension of preprocessed data.
The preprocessing assumes that images are 28 * 28 pixels, which covers all the images which are part of EMNIST dataset.

### 2. QCNN Circuit
**"QCNN_circuit.py"**: QCNN function implements Quantum Convolutional Neural Network.
"QCNN_structure" is standard structure with iterations of convolution and pooling layers.
"QCNN_structure_without_pooling" has only convolution layers.
"QCNN_1D_circuit" has connection between first and last qubits removed.
(**"Hierarchical.py"**: similarily for Hierarchical Quantum Classifier structure)
It has support for 9 Convolution and 3 pooling Ansatz to build QCNN.

**"unitary.py"**: contains all the unitary ansatz used for convolution and pooling layers.

**"embedding.py"**: shows how classical data is embedded for 8 qubit Quantum Convolutional Neural Network circuit.
There are five main embeddings: Amplitude Embedding, Angle Embedding, Compact Angle Embedding ("Angle-compact"), Hybrid Direct Embedding ("Amplitude-Hybrid"), Hybrid Angle Embedding ("Angulr-Hybrid").

**"Angular_hybrid.py"**: methods of Hybrid Angle Embedding 3 bits into 2 qubits and 15 bits into 4 qubits using Pauli , rotaion , controlled rotation, controlled NOT gates. 
Hybrid Direct Embedding and Hybrid Angle Embedding have variations depending on number of qubits in a embedding block and embedding arrangements. For example, "Amplitude-Hybrid4-1" embeds 16 classical data in 4 qubits in arrangement [[0,1,2,3], [4,5,6,7]].

### 3. QCNN Training
**"Training.py"**: trains quantum circuit (QCNN or Hierarchical). By default, it uses nesterov momentum optimizer to train 200 iteartions with batch size of 25. Both MSE Loss and cross entropy loss can be used for circuit training. Number of total parameters (total_params) need to be adjusted when testing different QCNN structures. 

### 4. Benchmarking
**"Benchmarking.py"**: trains quantum circuit for given dataset, unitary ansatze, and encoding / embedding method. Saves training loss history and test data accuracy after training. Encoding_to_Embedding function converts Encoding (classical preprocessing feature reduction) to Embedding (Classical data embedding into Quantum Circuit).


Binary: "True" uses 1 and -1 labels, while "False" uses 1 and 0 labels. When using cross entropy cost function always use "False".
When using mse cost function "True" in result for paper, but "False" can also be used.