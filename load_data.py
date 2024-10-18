# load_data.py

from sklearn.datasets import fetch_openml

def load_mnist():
    """
    Loads the MNIST dataset from openml and returns it in a dictionary format.
    The data is normalized and reshaped.
    
    Returns:
        dict: A dictionary with 'data' and 'target' keys, where:
            - 'data': contains the image data as a 2D array (n_samples, n_features)
            - 'target': contains the labels corresponding to each image
    """
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data / 255.0  # Normalize pixel values between 0 and 1
    target = mnist.target.astype(int)  # Convert targets to integers
    
    return {'data': data, 'target': target}
