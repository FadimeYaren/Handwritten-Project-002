Visit the website for results of model and instructions for being able to test it manually with your own handwritten digits.
# https://fadimeyaren.github.io/Handwritten-Project-002/
Handwritten-Project-002

Model Structure

Input → Hidden1 → Hidden2 → Hidden3 → Output

# Input 
(1x28x28)

# Hidden Layer 1
Conv2d(1→32, 3x3, pad=1)
ReLU()
MaxPool2d(2x2)

# Hidden Layer 2
Conv2d(32→64, 3x3, pad=1)
ReLU()
MaxPool2d(2x2)

# Hidden Layer 3
Dropout(p=0.3)
Flatten()
FC1+ReLU

# Output
FC2 (Linear → logits)
