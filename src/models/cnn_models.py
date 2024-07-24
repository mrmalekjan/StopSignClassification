from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

def CNN_model_1(input_shape = (64,64,3), num_classes=58):
    
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=input_shape))
    model.add(MaxPool2D())

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

if __name__ == "__main__":
    model = CNN_model_1()
    model.summary()