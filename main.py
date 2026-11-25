import random

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        # Initialize weights and bias with small random values
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate

    def _activate(self, sum_val):
        """Step activation function for binary classification."""
        return 1 if sum_val >= 0 else 0

    def predict(self, inputs):
        """Calculates the weighted sum of inputs and applies the activation function."""
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self._activate(weighted_sum)

    def train(self, training_data, epochs=100):
        """Trains the perceptron using the perceptron learning rule."""
        print("Training Perceptron...")
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction # Calculate the error
                total_error += abs(error)

                # Update weights and bias based on the error to learn from mistakes
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error
            if total_error == 0:
                # If no errors occurred in an epoch, the model has learned perfectly
                break
        print("Training complete.")

# --- Example Usage: Demonstrating an AI Security Challenge ---
if __name__ == "__main__":
    # Define a simple dataset for binary classification (2D points).
    # We aim to classify points in the 'top-right' region (e.g., x>0.7, y>0.7) as 1, others as 0.
    # This creates a linearly separable problem suitable for a perceptron.
    training_data = [
        ([0.1, 0.1], 0), # Bottom-left, expected 0
        ([0.1, 0.9], 0), # Top-left, expected 0
        ([0.9, 0.1], 0), # Bottom-right, expected 0
        ([0.9, 0.9], 1), # Top-right, expected 1 (positive class)
        ([0.2, 0.2], 0),
        ([0.8, 0.8], 1),
        ([0.3, 0.7], 0),
        ([0.7, 0.3], 0),
        ([0.5, 0.5], 0), # Another point expected 0
        ([0.75, 0.75], 1) # Another point expected 1
    ]

    # Initialize and train the perceptron model
    perceptron = Perceptron(num_inputs=2, learning_rate=0.05)
    perceptron.train(training_data, epochs=1000)

    print("\n--- Demonstrating AI Security Challenge ---")

    # 1. Test with a 'clean' input that should be classified correctly (expected 0)
    # This point is near the decision boundary but should be on the '0' side.
    original_input = [0.6, 0.6]
    original_prediction = perceptron.predict(original_input)
    print(f"Original input (expected 0): {original_input} -> Predicted: {original_prediction}")

    # 2. Create a slightly perturbed version of the original input.
    # This perturbation is small, mimicking an 'imperceptible' change in real-world scenarios.
    # The goal is to push the input across the model's decision boundary, causing misclassification.
    perturbation_amount = 0.08 # A small, carefully chosen change
    adversarial_input = [original_input[0] + perturbation_amount, original_input[1] + perturbation_amount]
    adversarial_prediction = perceptron.predict(adversarial_input)

    print(f"Adversarial input (original + {perturbation_amount} perturbation): {adversarial_input} -> Predicted: {adversarial_prediction}")

    # --- Analyze the result to highlight the AI security vulnerability ---
    # If the prediction changed from 0 to 1, the adversarial attack was successful.
    if original_prediction == 0 and adversarial_prediction == 1:
        print("\nAI Security Challenge Demonstrated: Small perturbation led to misclassification!")
        print("This highlights how AI models can be vulnerable to 'adversarial attacks' where")
        print("subtle, often imperceptible changes to input data can trick the system.")
        print("This requires specific AI security strategies, unlike traditional cybersecurity approaches.")
    else:
        print("\nAdversarial attack did not result in misclassification for this specific perturbation.")
        print("The effectiveness depends on the trained model's decision boundary and the perturbation.")
        print("Try adjusting 'original_input' or 'perturbation_amount' for a clearer demonstration.")

    # Optional: Test another clean input (expected 1) to show general model behavior
    clean_input_positive = [0.85, 0.85]
    clean_prediction_positive = perceptron.predict(clean_input_positive)
    print(f"\nClean input {clean_input_positive} -> Predicted: {clean_prediction_positive} (Expected: 1)")
