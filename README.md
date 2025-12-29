# ECG Signal Segmentation and Arrhythmia Detection

This project implements an automated system for ECG signal segmentation using the Viterbi Algorithm (Dynamic Programming) and Arrhythmia Classification using a Random Forest Classifier. It features a user-friendly Graphical User Interface (GUI) for visualizing signals, segmentation results, and diagnostic outputs.

## 📂 Repository Structure

The project is organized as follows:

- **Codes/**: Contains the main source code, including the GUI application and data processing scripts.
- **ML Model/**: Contains the Jupyter Notebooks used for training the Machine Learning model.
- **Data/**: Stores the dataset files (MIT-BIH Database).

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.x installed. You will need the following libraries:
```bash
pip install pandas numpy matplotlib scikit-learn
```

**Note:** The system also requires `tkinter`, which is usually included with Python.

### 2. Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/AbdelMoety/ECG-Classifier.git
cd ECG-Classifier
```

### 3. Setup

Before running the GUI, ensure that the trained model (`RFC_model.pkl`) is available.

- If the model file is not present, run the notebook in `ML Model/ML.ipynb` to generate it.
- **Important:** Place the `RFC_model.pkl` file in the same directory as `GUI.py` (inside the `Codes/` folder) for the application to load it correctly.

## 🖥️ How to Run the Code

The entire application is controlled via the GUI. To start the program:

1. Navigate to the Codes directory:
```bash
   cd Codes
```

2. Run the `GUI.py` file:
```bash
   python GUI.py
```

3. **Using the GUI:**
   - Click "Load Signal CSV" to select a patient's ECG file.
   - The system will automatically:
     - Visualize the Raw Signal.
     - Perform Signal Processing (Noise Removal).
     - Segment the waves (P, QRS, T) using the Viterbi Algorithm.
     - Classify the beat as Normal or Abnormal.
   - Results and medical metrics (QRS Width, PR Interval) will be displayed on the screen.

## 🛠️ Technologies Used

- **Python**: Core programming language.
- **Tkinter**: GUI framework.
- **Scikit-Learn**: Random Forest implementation for classification.
- **NumPy & Pandas**: Data manipulation and feature extraction.
- **Matplotlib**: Signal plotting and visualization.

## 👥 Authors (Team 6)

- Mostafa Hany Tawfik
- Ahmed Abdel Moety
- Alaa Essam
- Karim Hassan
- Omar Ahmed
