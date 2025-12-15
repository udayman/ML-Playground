import numpy as np

def init_params(input_size, hidden_size, output_size):
    """
    Initialize parameters for a 2-layer MLP.
    W1: (input_size, hidden_size)
    b1: (hidden_size,)
    W2: (hidden_size, output_size)
    b2: (output_size,)
    """
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)
    
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def forward(params, x):
    """
    Forward pass of the 2-layer MLP.
    x: (batch_size, input_size) or (input_size,)
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    
    # Layer 1
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    
    # Layer 2
    z2 = np.dot(a1, W2) + b2
    # Output is linear for now (e.g. regression)
    output = z2
    
    cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2}
    return output, cache

def backward(params, cache, grad_output):
    """
    Backward pass.
    grad_output: Gradient of loss w.r.t output (dL/dz2)
    """
    W2 = params['W2']
    a1 = cache['a1']
    z1 = cache['z1']
    x = cache['x'] # Input to the network
    
    # Gradients for Layer 2
    # z2 = a1 @ W2 + b2
    # dL/dW2 = a1.T @ dL/dz2
    dW2 = np.dot(a1.T, grad_output)
    db2 = np.sum(grad_output, axis=0) # Sum over batch dimension
    
    # Gradients for Layer 1
    # dL/da1 = dL/dz2 @ W2.T
    da1 = np.dot(grad_output, W2.T)
    # dL/dz1 = dL/da1 * relu'(z1)
    dz1 = da1 * relu_derivative(z1)
    
    dW1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

if __name__ == "__main__":
    # Test the implementation
    input_size = 10
    hidden_size = 20
    output_size = 5
    batch_size = 4
    
    params = init_params(input_size, hidden_size, output_size)
    
    # Random input and target
    x = np.random.randn(batch_size, input_size)
    y_true = np.random.randn(batch_size, output_size)
    
    print("Initial parameters initialized.")
    
    # Forward pass
    output, cache = forward(params, x)
    print("Forward pass output shape:", output.shape)
    
    # Loss (MSE) = 0.5 * mean((y_pred - y_true)^2)
    loss = 0.5 * np.mean((output - y_true)**2)
    print(f"Initial Loss: {loss:.4f}")
    
    # Backward pass
    # dL/dOutput = (y_pred - y_true) / batch_size
    grad_output = (output - y_true) / batch_size
    
    grads = backward(params, cache, grad_output)
    print("Gradients computed.")
    
    # Update parameters
    learning_rate = 0.1
    for k in params:
        params[k] -= learning_rate * grads[k]
        
    print("Parameters updated.")
    
    # Check loss after update
    output_new, _ = forward(params, x)
    loss_new = 0.5 * np.mean((output_new - y_true)**2)
    print(f"Loss after 1 step: {loss_new:.4f}")
    
    if loss_new < loss:
        print("SUCCESS: Loss decreased.")
    else:
        print("FAILURE: Loss did not decrease.")
