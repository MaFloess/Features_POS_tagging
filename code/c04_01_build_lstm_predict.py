# Imports
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM



# One-hot-encoding for a single array
def one_hot_encoding(arr):
    pred = np.argmax(arr)
    result = np.array(range(len(arr)))
    
    np.putmask(result, result == pred, 1000)
    np.putmask(result, result < 1000, 0)
    np.putmask(result, result == 1000, 1)
        
    return result



# One-hot-encoding for a single sentence
def one_hot_pred(softmax_output):
       
    sent = []
            
    for i in range(softmax_output.shape[1]):        
        sent.append(one_hot_encoding(softmax_output[0][i]))
            
    return np.array(sent, dtype=object)



# Build a LSTM Neural Network (POS-tagging model) and make predictions
def build_lstm_predict(X_train, Y_train, X_dev, unit_count):

	# Setup model
    m = Sequential()
    m.add(LSTM(units=unit_count, return_sequences=True))
    m.add(Dense(units=Y_train.shape[2], activation='softmax'))
    m.compile(optimizer='adam', loss='categorical_crossentropy')

    # Train model
    m.fit(X_train, Y_train, epochs=25, batch_size=100)

    # Make list of sentence wise predictions one-hot-encoded
    predictions = []

    for sent in X_dev:
        predictions.append(one_hot_pred(m.predict(sent.reshape(1, sent.shape[0], sent.shape[1]))))

    return predictions