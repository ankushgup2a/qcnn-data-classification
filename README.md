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

Circuit image with 8 qubits, Ansatz = 'U_SU4', U_params = 15, Encoding = 'AmplitudeEncoding',circuit = 'QCNN',cost_fn = 'cross_entropy'

0: ─╭AmplitudeEmbedding(M0)──U3(-0.45,0.12,-1.50)──╭●──RY(-1.21)─╭X──RY(0.56)─╭●
1: ─├AmplitudeEmbedding(M0)────────────────────────│─────────────│────────────│─
2: ─├AmplitudeEmbedding(M0)────────────────────────│─────────────│────────────│─
3: ─├AmplitudeEmbedding(M0)────────────────────────│─────────────│────────────│─
4: ─├AmplitudeEmbedding(M0)────────────────────────│─────────────│────────────│─
5: ─├AmplitudeEmbedding(M0)────────────────────────│─────────────│────────────│─
6: ─├AmplitudeEmbedding(M0)────────────────────────│─────────────│────────────│─
7: ─╰AmplitudeEmbedding(M0)──U3(-0.65,-0.01,-1.23)─╰X──RZ(-0.17)─╰●───────────╰X

───U3(0.05,0.63,1.15)─────U3(-0.45,0.12,-1.50)──╭●──────────RY(-1.21)─╭X─────────RY(0.56)
───U3(-0.65,-0.01,-1.23)────────────────────────╰X──────────RZ(-0.17)─╰●─────────────────
───U3(-0.45,0.12,-1.50)──╭●──────────────────────RY(-1.21)─╭X──────────RY(0.56)─╭●───────
───U3(-0.65,-0.01,-1.23)─╰X──────────────────────RZ(-0.17)─╰●───────────────────╰X───────
───U3(-0.45,0.12,-1.50)──╭●──────────────────────RY(-1.21)─╭X──────────RY(0.56)─╭●───────
───U3(-0.65,-0.01,-1.23)─╰X──────────────────────RZ(-0.17)─╰●───────────────────╰X───────
───U3(-0.45,0.12,-1.50)─────────────────────────╭●──────────RY(-1.21)─╭X─────────RY(0.56)
───U3(-0.71,1.19,-0.65)───U3(-0.65,-0.01,-1.23)─╰X──────────RZ(-0.17)─╰●─────────────────

──╭●─────────────────────U3(0.05,0.63,1.15)────────────────────────────────────────────────
──╰X─────────────────────U3(-0.71,1.19,-0.65)───U3(-0.45,0.12,-1.50)──╭●──────────RY(-1.21)
───U3(0.05,0.63,1.15)────U3(-0.65,-0.01,-1.23)────────────────────────╰X──────────RZ(-0.17)
───U3(-0.71,1.19,-0.65)──U3(-0.45,0.12,-1.50)──╭●──────────────────────RY(-1.21)─╭X────────
───U3(0.05,0.63,1.15)────U3(-0.65,-0.01,-1.23)─╰X──────────────────────RZ(-0.17)─╰●────────
───U3(-0.71,1.19,-0.65)──U3(-0.45,0.12,-1.50)─────────────────────────╭●──────────RY(-1.21)
──╭●─────────────────────U3(0.05,0.63,1.15)─────U3(-0.65,-0.01,-1.23)─╰X──────────RZ(-0.17)
──╰X─────────────────────U3(-0.71,1.19,-0.65)──────────────────────────────────────────────

──────────────────────────────────────────────────────────────────╭RZ(-0.99)────╭RX(3.11)
──╭X─────────RY(0.56)─╭●─────────────────────U3(0.05,0.63,1.15)───╰●──────────X─╰●───────
──╰●──────────────────╰X─────────────────────U3(-0.71,1.19,-0.65)─╭RZ(-0.99)────╭RX(3.11)
───RY(0.56)─╭●─────────U3(0.05,0.63,1.15)─────────────────────────╰●──────────X─╰●───────
────────────╰X─────────U3(-0.71,1.19,-0.65)───────────────────────╭RZ(-0.99)────╭RX(3.11)
──╭X─────────RY(0.56)─╭●─────────────────────U3(0.05,0.63,1.15)───╰●──────────X─╰●───────
──╰●──────────────────╰X─────────────────────U3(-0.71,1.19,-0.65)─╭RZ(-0.99)────╭RX(3.11)
──────────────────────────────────────────────────────────────────╰●──────────X─╰●───────

───U3(-0.49,0.25,0.40)──╭●──RY(-0.68)─╭X──RY(-0.67)─╭●──U3(-0.14,0.02,-1.36)──U3(-0.49,0.25,0.40)─
────────────────────────│─────────────│─────────────│─────────────────────────────────────────────
────────────────────────│─────────────│─────────────│───U3(-0.04,0.47,-0.15)──────────────────────
────────────────────────│─────────────│─────────────│─────────────────────────────────────────────
────────────────────────│─────────────│─────────────│───U3(-0.49,0.25,0.40)───────────────────────
────────────────────────│─────────────│─────────────│─────────────────────────────────────────────
───U3(-0.04,0.47,-0.15)─╰X──RZ(1.26)──╰●────────────╰X──U3(0.19,0.63,-1.82)───U3(-0.04,0.47,-0.15)
──────────────────────────────────────────────────────────────────────────────────────────────────

──╭●──RY(-0.68)─╭X──RY(-0.67)─╭●──U3(-0.14,0.02,-1.36)───────────────────────────────────────
──│─────────────│─────────────│──────────────────────────────────────────────────────────────
──╰X──RZ(1.26)──╰●────────────╰X──U3(0.19,0.63,-1.82)───U3(-0.49,0.25,0.40)──╭●──RY(-0.68)─╭X
─────────────────────────────────────────────────────────────────────────────│─────────────│─
──╭●──RY(-0.68)─╭X──RY(-0.67)─╭●──U3(-0.14,0.02,-1.36)──U3(-0.04,0.47,-0.15)─╰X──RZ(1.26)──╰●
──│─────────────│─────────────│──────────────────────────────────────────────────────────────
──╰X──RZ(1.26)──╰●────────────╰X──U3(0.19,0.63,-1.82)────────────────────────────────────────
─────────────────────────────────────────────────────────────────────────────────────────────

──────────────────────────────────────╭RZ(-2.15)────╭RX(0.71)──U3(-0.81,0.13,-0.76)─╭●──RY(-0.41)─╭X
──────────────────────────────────────│─────────────│───────────────────────────────│─────────────│─
───RY(-0.67)─╭●──U3(-0.14,0.02,-1.36)─╰●──────────X─╰●──────────────────────────────│─────────────│─
─────────────│──────────────────────────────────────────────────────────────────────│─────────────│─
─────────────╰X──U3(0.19,0.63,-1.82)──╭RZ(-2.15)────╭RX(0.71)──U3(1.27,-1.54,0.76)──╰X──RZ(-0.66)─╰●
──────────────────────────────────────│─────────────│───────────────────────────────────────────────
──────────────────────────────────────╰●──────────X─╰●──────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────────────────────────

───RY(1.54)─╭●──U3(-0.02,0.72,0.02)──╭●──────────X─╭●────────┤       
────────────│────────────────────────│─────────────│─────────┤       
────────────│────────────────────────│─────────────│─────────┤       
────────────│────────────────────────│─────────────│─────────┤       
────────────╰X──U3(-0.81,0.96,-0.06)─╰RZ(-2.16)────╰RX(0.60)─┤  Probs
─────────────────────────────────────────────────────────────┤       
─────────────────────────────────────────────────────────────┤       
─────────────────────────────────────────────────────────────┤       