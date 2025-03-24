# 📰 Fake News Detector

## 📌 Overview
The **Fake News Detector** is a machine learning project that classifies news articles as **real** or **fake** using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. The project leverages **sentence embeddings** and a **Logistic Regression** model to analyze textual data and provide accurate classifications.

## 🚀 Features
- ✅ **Detects fake and real news articles** with high accuracy
- ✅ Utilizes **Sentence Transformers** for text embedding
- ✅ **Preprocesses text** using NLP techniques (cleaning, tokenization, etc.)
- ✅ **Trains and evaluates a Logistic Regression model**
- ✅ **Visualizes performance** using a confusion matrix

## 📊 Dataset
This project uses a labeled dataset of real and fake news articles. The data undergoes **preprocessing**, including:
- **Text cleaning** (removal of special characters, conversion to lowercase)
- **Feature extraction** using **sentence embeddings** (`all-MiniLM-L6-v2`)
- **Splitting into training and testing sets**

## 🛠️ Technologies Used
- 🐍 **Python**
- 📒 **Jupyter Notebook**
- 🤖 **Scikit-learn**
- 📊 **Pandas & NumPy**
- 🗣️ **Sentence Transformers** (`all-MiniLM-L6-v2`)
- 📈 **Matplotlib** (for result visualization)

## 📌 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/LuukaSuresh/Fake-News-Detector.git
   cd Fake-News-Detector
   ```
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn sentence-transformers joblib matplotlib
   ```
3. Run the Jupyter Notebook or Python script to train and evaluate the model.

## 📌 Usage
1. Load the dataset (`Fake.csv`, `True.csv`).
2. Preprocess the data (cleaning and embedding generation).
3. Train the **Logistic Regression** model.
4. Evaluate the model using **accuracy score and confusion matrix**.

## 📌 Future Improvements
- 🚀 Improve accuracy with **advanced deep learning models** (LSTMs, Transformers)
- 📊 Develop a **web-based UI** for easier interaction
- 🗂️ Expand dataset for **better generalization**

## 👨‍💻 Contributing
Contributions are welcome! Feel free to **open issues** and submit **pull requests**.

## 📞 Contact
For any inquiries, reach out via **[sureshmadushan0623@gmail.com](mailto:sureshmadushan0623@gmail.com)**  
💻 **GitHub**: [LuukaSuresh](https://github.com/LuukaSuresh)

