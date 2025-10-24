# 🚨 Cyberbullying Detection Using Deep Learning

This project focuses on detecting **cyberbullying in social media platforms** using **Natural Language Processing (NLP)** and **Deep Learning models**.  
We used **RoBERTa**, a transformer-based language model, to classify online text (tweets, comments, and posts) into **bullying** or **non-bullying** categories.  
Additionally, we extended the system to analyze **text in images (memes)** using OCR, making the model more robust for real-world scenarios.

---

## 🧠 Project Goal
- Detect and classify cyberbullying text from multiple datasets (Main, Reddit, Tweets).  
- Build a deep learning classification model using `RoBERTa`.  
- Visualize results through metrics like accuracy, precision, recall, and F1-score.  
- Extract and analyze text from meme images using OCR.

---

## 📊 Datasets Used
1. **Main Dataset** – Labeled bullying and non-bullying posts.  
2. **Reddit Dataset** – Real comments collected from Reddit.  
3. **Tweets Dataset** – Tweets labeled as cyberbullying or not.  

| Dataset | Size  | Classes          |
|---------|-------|-------------------|
| Main    | 8,452 | Bullying / Non-Bullying |
| Reddit  | 5,966 | Bullying / Non-Bullying |
| Tweets  | 47,692| Bullying / Non-Bullying |

---

## 🧰 Tech Stack
- Python 🐍  
- Jupyter Notebook  
- Transformers (HuggingFace)  
- Scikit-learn  
- Pandas, NumPy, Matplotlib, Seaborn  
- Tesseract OCR (for image text extraction)  
- PIL, OpenCV

---

## 🧪 Model Details
- **Base Model:** `RoBERTa`  
- **Tokenizer:** `RobertaTokenizerFast`  
- **Training:** Fine-tuned on bullying datasets  
- **Metrics:** Accuracy, Precision, Recall, F1 Score  
- **Extra:** Meme text extraction using OCR

---

## 📈 Performance Summary

| Dataset | Accuracy | F1 Score | Precision | Recall |
|---------|----------|----------|-----------|--------|
| Main    | 93.7%    | 94.5%    | 94.9%     | 94.0%  |
| Reddit  | 91.2%    | 92.3%    | 91.8%     | 92.0%  |
| Tweets  | 95.0%    | 95.3%    | 95.1%     | 95.2%  |

---

## 🖼️ Sample Image Detection
The project also includes meme image text extraction and classification.

```python
# Example
Text Extracted: "You're such a loser 😡"
Prediction: Bullying ✅
