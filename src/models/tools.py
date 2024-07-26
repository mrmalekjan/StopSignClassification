import tensorflow as tf
import os

def load_model(model_name):
    recent_folder = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(recent_folder, 'trained_models', model_name)
    print(load_path)
    model = tf.keras.models.load_model(load_path)
    print(f'{model_name} loaded seccessfully!')
    return model
    
def save_model(model, model_name):
    recent_folder = os.path.dirname(os.path.abspath(__file__))
    model.save(os.path.join(recent_folder, 'trained_models', model_name))
    print(f'{model_name} saved seccessfully!')
    return
        
