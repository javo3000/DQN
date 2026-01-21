import numpy as np
import matplotlib.pyplot as plt
import time

epochs = 10000
learning_rate = 0.01
target_safety_value = 0.89

# History tracking for visualization
history = {
    'epoch': [],
    'prediction': [],
    'error': [],
    'grad_w3': [],
    'grad_w2': [],
    'grad_w1': []
}

class safety_value_network:
    def __init__(self):
            self.weights = np.random.randn(133, 256) * 0.01
            self.weights2 = np.random.randn(256, 256) * 0.01
            self.weights3 = np.random.randn(256, 1) * 0.01  # Output single scalar
            self.bias = np.zeros(256)
            self.bias2 = np.zeros(256)
            self.bias3 = np.zeros(1)  # Scalar bias for final output
    def relu(self, x):
        return np.maximum(0, x)
        #here is the rectifier function
        #return x if x > 0 else 0
        #above is another way to write the same function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def forward_pass1(self, state):
        a1 = np.dot(state, self.weights) + self.bias
        self.z1 = a1
        score = self.relu(a1)
        return score
    def forward_pass2(self, score):
        a2 = np.dot(score, self.weights2) + self.bias2
        value = self.relu(a2)
        self.z2 = a2
        return value
    def forward_pass3(self, value):
        a3 = np.dot(value, self.weights3) + self.bias3
        self.z3 = a3
        safety_value = self.sigmoid(a3)  # Squash to [0, 1]
        return safety_value.item()

    def forward_pass(self, state, target):
        score = self.forward_pass1(state)
        value = self.forward_pass2(score)
        safety_value = self.forward_pass3(value)
        self.a1 = score
        self.a2 = value
        self.a3 = safety_value
        self.state = state
        self.backward_pass(state, target)
        return safety_value
        

    def relu_derivative(self, x):
        #return 1 if x > 0 else 0
        #another way to write the same function
        return np.where(x > 0, 1, 0)
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def backward_pass(self, state, target):
        #calculate the error
        # 1. Calculate the error at the output layer
        error = self.a3 - target
        self.error1 = error

        # 2. Compute the gradient of the loss with respect to the output layer's pre-activation (z3)
        # Using the chain rule: dL/dz3 = (dL/da3) * (da3/dz3)
        d_lw3 = error #there is a beutiful simplification for the binary cross-entropy loss that allows this statement

        # 3. Compute the gradient of the loss with respect to the second hidden layer's activation (a2)
        # dL/da2 = (dL/dz3) * (dz3/da2)
        d_a2 = np.dot(d_lw3, self.weights3.T)
        d_b3 = d_lw3
        self.grad_w3 = np.dot(self.a2.T, d_lw3)
        self.weights3 -= learning_rate * self.grad_w3
        self.bias3 -= learning_rate * d_b3
        self.backward_pass2(d_a2)
        pass

    def backward_pass2(self, d_a2):
        d_lw2 = d_a2 * self.relu_derivative(self.z2)
        d_a1 = np.dot(d_lw2, self.weights2.T)
        d_b2 = np.sum(d_lw2, axis=0)  # Flatten to match bias shape
        self.grad_w2 = np.dot(self.a1.T, d_lw2)
        self.weights2 -= learning_rate * self.grad_w2
        self.bias2 -= learning_rate * d_b2
        self.backward_pass3(d_a1)
        pass 
    def backward_pass3(self, d_a1):
        d_lw1 = d_a1 * self.relu_derivative(self.z1)
        d_state = np.dot(d_lw1, self.weights.T)
        self.grad_w1 = np.dot(self.state.T, d_lw1)
        d_b1 = np.sum(d_lw1, axis=0)  # Flatten to match bias shape
        self.weights -= learning_rate * self.grad_w1
        self.bias -= learning_rate * d_b1
        pass

state = np.random.randn(1, 133)
model = safety_value_network()

# Start timing
start_time = time.time()

for current_epoch in range(1, epochs + 1):
    model.forward_pass(state, target_safety_value)
    
    # Record history every 100 epochs for smooth plotting
    if current_epoch % 100 == 0:
        history['epoch'].append(current_epoch)
        history['prediction'].append(model.a3)
        history['error'].append(model.error1)
        history['grad_w3'].append(np.linalg.norm(model.grad_w3))
        history['grad_w2'].append(np.linalg.norm(model.grad_w2))
        history['grad_w1'].append(np.linalg.norm(model.grad_w1))
    
    # Print every 1000 epochs
    if current_epoch % 1000 == 0:
        print(f"Epoch: {current_epoch}, Prediction: {model.a3:.6f}, Error: {model.error1:.6f}, "
              f"Grad W3: {np.linalg.norm(model.grad_w3):.6f}, "
              f"Grad W2: {np.linalg.norm(model.grad_w2):.6f}, "
              f"Grad W1: {np.linalg.norm(model.grad_w1):.6f}")

# End timing
training_time = time.time() - start_time
print(f"\n⏱️  Training completed in {training_time:.3f} seconds")
print(f"   Throughput: {epochs/training_time:.0f} epochs/sec")
print(f"   Time per epoch: {training_time/epochs*1000:.3f} ms")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Safety Value Network - Training Dynamics', fontsize=14, fontweight='bold')

# Plot 1: Prediction vs Target
axes[0, 0].plot(history['epoch'], history['prediction'], 'b-', linewidth=2, label='Prediction')
axes[0, 0].axhline(y=target_safety_value, color='r', linestyle='--', linewidth=2, label=f'Target ({target_safety_value})')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Safety Value V(x)')
axes[0, 0].set_title('Prediction Convergence')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Error over time
axes[0, 1].plot(history['epoch'], history['error'], 'orange', linewidth=2)
axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Error (prediction - target)')
axes[0, 1].set_title('Error Convergence')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Gradient norms
axes[1, 0].plot(history['epoch'], history['grad_w3'], label='∇W3 (output)', linewidth=2)
axes[1, 0].plot(history['epoch'], history['grad_w2'], label='∇W2 (hidden)', linewidth=2)
axes[1, 0].plot(history['epoch'], history['grad_w1'], label='∇W1 (input)', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Gradient Norm')
axes[1, 0].set_title('Gradient Flow by Layer')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Absolute error (loss proxy)
axes[1, 1].plot(history['epoch'], np.abs(history['error']), 'green', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('|Error|')
axes[1, 1].set_title('Absolute Error (Loss Proxy)')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')

print(f"\nFinal Results:")
print(f"  Target: {target_safety_value}")
print(f"  Prediction: {model.a3:.6f}")
print(f"  Error: {model.error1:.6f}")
print(f"\n📊 Learning curve saved to: learning_curve.png")

plt.show(block=False)
plt.pause(3)  # Show for 3 seconds then continue
plt.close()
