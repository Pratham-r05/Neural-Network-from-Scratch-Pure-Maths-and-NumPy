# 🧠 Building a Neural Network from Scratch

Welcome to the **Building_NN_from_aScratch** project!  
This repository demonstrates building a simple neural network from the ground up using only `numpy` and `pandas`. The notebook walks through all steps needed to train a model on the MNIST-like digit dataset, perform inference, and visualize predictions—without any deep learning libraries!

---

## 🌟 Project Highlights

- **Zero Frameworks:** No TensorFlow or PyTorch—just math!
- **Step-by-Step:** All neural net logic is implemented and explained.
- **Educational:** Great for students and beginners.
- **Visualization:** See the network’s predictions as images.

---

## 🚀 Quick Start

1. Clone the repo:
   ```bash
   git clone https://github.com/Pratham-r05/Projects_2.git
   cd Projects_2
   ```

2. Open the notebook:
   ```
   Building_NN_from_aScratch (1).ipynb
   ```

3. Make sure you have the required libraries:
   ```bash
   pip install numpy pandas matplotlib
   ```

4. Add your dataset (`train.csv`) to the root directory.

5. Run the notebook step-by-step! 📓

---

## 📝 How It Works

### 1. **Data Preparation**
- The dataset is loaded and shuffled for training and development splits.
- Input images are normalized.

```python
data = pd.read_csv('train.csv')
# ...shuffling and normalization...
```

### 2. **Network Architecture**
- Input layer: 784 neurons (28x28 images)
- Hidden layer: 10 neurons (ReLU activation)
- Output layer: 10 neurons (softmax activation)

### 3. **Forward and Backward Propagation**
- All math for ReLU, softmax, loss, and gradients is implemented by hand.

### 4. **Training**
- Gradient descent is run for 500 iterations with performance logged every 10 steps.

<p align="center">
  <img src="assets/training_accuracy.gif" alt="Training Accuracy Over Time" width="500"/>
</p>

## Training Accuracy

![Training Accuracy](/accuracy.png)
The plot above shows the training accuracy over epochs.

---

## 🧩 Code Structure

- `Building_NN_from_aScratch (1).ipynb` — Main notebook
- `train.csv` — Training dataset (not included for copyright)

---

## 📊 Example Output

Below is an example of how predictions are visualized:

```python
def test_prediction(index, W1, b1, W2, b2):
    # ...code...
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
```

![Example 1 Digit Prediction](/Screenshot (3).png)
![Example 2 Digit Prediction](/Screenshot (4).png)
![Example 3 Digit Prediction](/Screenshot (5).png)
![Example 4 Digit Prediction](/Screenshot (6).png)

---

## 🤝 Contributing

Feel free to fork, open issues, or submit pull requests if you improve the code or add new features!

---

## 📬 Contact

Created by [Pratham-r05](https://github.com/Pratham-r05).  
Open an issue for questions or feedback!

---

## ⭐ Star this repo if you found it helpful!
