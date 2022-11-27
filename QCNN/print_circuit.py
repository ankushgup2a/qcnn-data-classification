import data
import Training
import QCNN_circuit
import Hierarchical_circuit
import numpy as np
import pennylane as qml

def Encoding_to_Embedding(Encoding):
    # Amplitude Embedding / Angle Embedding
    if Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    elif Encoding == 'autoencoder8':
        Embedding = 'Angle'

    # Amplitude Hybrid Embedding
    # 4 qubit block
    elif Encoding == 'pca32-1':
        Embedding = 'Amplitude-Hybrid4-1'
    elif Encoding == 'autoencoder32-1':
        Embedding = 'Amplitude-Hybrid4-1'

    elif Encoding == 'pca32-2':
        Embedding = 'Amplitude-Hybrid4-2'
    elif Encoding == 'autoencoder32-2':
        Embedding = 'Amplitude-Hybrid4-2'

    elif Encoding == 'pca32-3':
        Embedding = 'Amplitude-Hybrid4-3'
    elif Encoding == 'autoencoder32-3':
        Embedding = 'Amplitude-Hybrid4-3'

    elif Encoding == 'pca32-4':
        Embedding = 'Amplitude-Hybrid4-4'
    elif Encoding == 'autoencoder32-4':
        Embedding = 'Amplitude-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca16-1':
        Embedding = 'Amplitude-Hybrid2-1'
    elif Encoding == 'autoencoder16-1':
        Embedding = 'Amplitude-Hybrid2-1'

    elif Encoding == 'pca16-2':
        Embedding = 'Amplitude-Hybrid2-2'
    elif Encoding == 'autoencoder16-2':
        Embedding = 'Amplitude-Hybrid2-2'

    elif Encoding == 'pca16-3':
        Embedding = 'Amplitude-Hybrid2-3'
    elif Encoding == 'autoencoder16-3':
        Embedding = 'Amplitude-Hybrid2-3'

    elif Encoding == 'pca16-4':
        Embedding = 'Amplitude-Hybrid2-4'
    elif Encoding == 'autoencoder16-4':
        Embedding = 'Amplitude-Hybrid2-4'

    # Angular HybridEmbedding
    # 4 qubit block
    elif Encoding == 'pca30-1':
        Embedding = 'Angular-Hybrid4-1'
    elif Encoding == 'autoencoder30-1':
        Embedding = 'Angular-Hybrid4-1'

    elif Encoding == 'pca30-2':
        Embedding = 'Angular-Hybrid4-2'
    elif Encoding == 'autoencoder30-2':
        Embedding = 'Angular-Hybrid4-2'

    elif Encoding == 'pca30-3':
        Embedding = 'Angular-Hybrid4-3'
    elif Encoding == 'autoencoder30-3':
        Embedding = 'Angular-Hybrid4-3'

    elif Encoding == 'pca30-4':
        Embedding = 'Angular-Hybrid4-4'
    elif Encoding == 'autoencoder30-4':
        Embedding = 'Angular-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca12-1':
        Embedding = 'Angular-Hybrid2-1'
    elif Encoding == 'autoencoder12-1':
        Embedding = 'Angular-Hybrid2-1'

    elif Encoding == 'pca12-2':
        Embedding = 'Angular-Hybrid2-2'
    elif Encoding == 'autoencoder12-2':
        Embedding = 'Angular-Hybrid2-2'

    elif Encoding == 'pca12-3':
        Embedding = 'Angular-Hybrid2-3'
    elif Encoding == 'autoencoder12-3':
        Embedding = 'Angular-Hybrid2-3'

    elif Encoding == 'pca12-4':
        Embedding = 'Angular-Hybrid2-4'
    elif Encoding == 'autoencoder12-4':
        Embedding = 'Angular-Hybrid2-4'

    # Two Gates Compact Encoding
    elif Encoding == 'pca16-compact':
        Embedding = 'Angle-compact'
    elif Encoding == 'autoencoder16-compact':
        Embedding = 'Angle-compact'
    return Embedding

# Define Constants/Variables
U = 'U_SU4'
U_params = 15
Encoding = 'resize256'
Embedding = Encoding_to_Embedding(Encoding)
circuit = 'QCNN'
cost_fn = 'cross_entropy'
classes = [0,1]
binary = False
dataset = 'mnist'

X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit, cost_fn)

x = X_test[0]
prediction = QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn)

print(qml.draw_mpl(QCNN_circuit.QCNN)(x, trained_params, U, U_params, Embedding, cost_fn))