# Smart Waste Classification using Deep Learning

## 📌 Problem Statement
Improper waste segregation reduces recycling efficiency and negatively impacts the environment.  
This project builds a deep learning-based image classification system to automatically categorize waste into six classes:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## 🗂 Dataset
Dataset used: TrashNet (publicly available waste image dataset)

- ~2500 images
- 6 material categories
- Images resized to 224x224
- 80% training / 20% validation split

---

## 🧠 Model Architecture

Transfer Learning approach using:

- **MobileNetV2 (Pretrained on ImageNet)**
- Global Average Pooling Layer
- Dense(128) + ReLU
- Dropout (0.3)
- Softmax output layer (6 classes)

The base model was initially frozen and later partially fine-tuned for improved performance.

---

## ⚙️ Training Details

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 32
- EarlyStopping applied
- Fine-tuning with reduced learning rate (1e-5)

---

## 📊 Results

- Validation Accuracy: ~75%
- Confusion matrix analysis performed
- Class-wise precision, recall, and F1-score evaluated

### Key Observation
Significant confusion was observed between **paper and cardboard** due to visual similarity in texture and color.

---

## 🔍 Key Learnings

- Transfer learning significantly improves performance on small datasets.
- Fine-tuning only higher layers stabilizes training.
- Visually similar material classes remain challenging for CNN models.

---

## 🚀 Future Improvements

- Increase dataset size
- Apply class balancing techniques
- Experiment with EfficientNet
- Deploy as a web application (Streamlit/Flask)
- Convert to TensorFlow Lite for mobile deployment

---

## 🛠 Requirements

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies:

```
pip install -r requirements.txt
```

---

## 📌 Project Structure

```
smart-waste-classification-deep-learning/
│
├── Smart_Waste_Classification.ipynb
├── README.md
└── requirements.txt
```

---

## 📎 Conclusion

This project demonstrates a practical application of deep learning and transfer learning for real-world environmental AI systems.
