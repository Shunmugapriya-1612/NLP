https://myfirstapppriya.streamlit.app/ - app

### 🎙️ **Sports Interview NLP Suite (BERT-AUG)**

This project aims to streamline the analysis of sports interview transcripts using state-of-the-art Natural Language Processing (NLP) models. It leverages BERT (Bidirectional Encoder Representations from Transformers) for **interview transcript classification**, and GPT-2 for generating AI-powered interview responses. Additionally, it provides interactive visualizations to explore topic clusters derived from the interview data.

#### Key Features:

* **Transcript Classification**: Uses a fine-tuned BERT model for classifying sports interview transcripts into predefined categories such as *pre-game*, *post-game reactions*, *injury updates*, and others. This is based on a custom classification model trained with augmented data using synonym replacement techniques.
* **AI Q\&A Generation**: GPT-2 is employed to generate interview responses based on the selected category and a user-entered question.
* **Interactive Visualization**: UMAP/t-SNE embeddings are visualized to explore the relationships and clusters among the categorized interview transcripts, aiding in deeper insights into topic distributions.

#### **Model Training & Augmentation**:

The classification model is based on BERT, which has been fine-tuned for this specific task. The following techniques have been applied:

1. **Synonym Replacement (Data Augmentation)**: To enhance model robustness, synonym replacement is applied to the training texts. This is done by randomly replacing words in the transcript with their synonyms using the NLTK WordNet.
2. **Text Preprocessing & Tokenization**: Texts are tokenized using the BERT tokenizer, which prepares the data for the BERT model by truncating, padding, and ensuring that the sequence length is consistent.
3. **Model Architecture**: The model is a `BertForSequenceClassification` from the Hugging Face `transformers` library, adapted for multi-class classification.
4. **Training Loop**: The model is trained with the AdamW optimizer, leveraging GPU acceleration if available, and a learning rate of `5e-5`.
5. **Evaluation Metrics**: The model performance is evaluated using **accuracy**, **weighted F1 score**, **classification report**, and **confusion matrix** to provide comprehensive insights into its effectiveness.

#### **Model Evaluation**:

The model was evaluated on a separate test set, yielding the following results:

* **Accuracy**: A high accuracy score demonstrating the model's reliability in categorizing interview transcripts.
* **F1 Score**: The weighted F1 score reflects the balance between precision and recall across different categories.
* **Classification Report**: Detailed performance metrics for each class, helping assess the model's performance in different transcript categories.
* **Confusion Matrix**: Visualized to better understand the model's misclassifications.

#### **App Functionality**:

1. **Transcript Classification**: Users can paste an interview transcript into a text area, and the app will predict the category of the transcript using the trained BERT classifier.
2. **Q\&A Generation**: Users select an interview category and enter a question. The app then generates a response using GPT-2, offering an AI-powered answer based on the context of the interview.
3. **Embedding Visualization**: A visualization tab allows users to explore the topic clusters of interview transcripts. UMAP/t-SNE methods are employed to reduce dimensionality and create a visual representation of the text embeddings.

---

### **Model and Application Workflow**

1. **Data Augmentation**:

   * Synonym replacement was applied to the interview transcripts to generate additional augmented data for training. This helps the model generalize better and handle different wordings of the same concept in interview texts.
2. **Data Preprocessing**:

   * Transcripts were cleaned and tokenized using the BERT tokenizer. The input text was processed with truncation and padding to ensure compatibility with BERT’s input requirements.
3. **Model Training**:

Model Classifier: has been uploaded in the Shunmugapriya1612/sports-bert-classifier repository

<img width="949" alt="image" src="https://github.com/user-attachments/assets/735f7286-cc5d-4d91-a001-45d4c1cc942b" />


   * The augmented dataset was split into training and testing sets, with an 80/20 split. The training loop included 25 epochs, with loss being computed at each step.
4. **Model Evaluation**:

   * After training, the model was evaluated on the test set. Key metrics such as accuracy, F1 score, and confusion matrix were used to assess its performance.
5. **Deployment**:

   * The final model and tokenizer are saved using `torch.save` and `joblib.dump`, ensuring that the trained components can be easily loaded for inference during deployment.

---

### **Technologies Used**:

* **Streamlit**: Interactive app interface to present model results, answer generation, and visualizations.
* **Transformers Library (Hugging Face)**: Utilized for pre-trained BERT models and the text-generation pipeline (GPT-2).
* **Torch**: The deep learning framework for model training, evaluation, and inference.
* **Plotly**: For interactive data visualization of the transcript embeddings.
* **NLTK**: For synonym replacement during data augmentation.

---

### **Getting Started**:

To run this project locally or on Streamlit Cloud:

1. Clone the repository.
2. Install dependencies from the `requirements.txt` file.
3. Ensure your system has the necessary models (`BERT`, `GPT-2`).
4. Run the Streamlit app using `streamlit run app.py`.

---

### **Future Improvements**:

* **Model Fine-Tuning**: Enhance the model’s performance by fine-tuning with more diverse sports-related transcripts.
* **Additional Augmentation Techniques**: Explore other data augmentation methods, such as back-translation or random word insertion.
* **Multilingual Support**: Extend the model to support interviews in multiple languages.

---

### **Example Outputs**:

#### **Transcript Classification**:

**Input Transcript**: *"The team has prepared really well for this match. We are confident going into this game."*

**Predicted Category**: *Pre-game*

#### **Generated Response**:

**Input Question**: *"How did you feel after the match?"*

**Generated AI Response**: *"It was a tough match, but the team played well. We're happy with our performance."*

**Features of StreamLit App**
---
**Classify Interview Transcript**

<img width="763" alt="image" src="https://github.com/user-attachments/assets/ac9f9f03-8ff3-49ba-a8fa-dd2c084dc9ed" />

---
**Generate AI Interview Response**

<img width="641" alt="image" src="https://github.com/user-attachments/assets/8da9d11f-c65b-44e1-9585-77d405b168b4" />

---
**Topic Clusters**

<img width="633" alt="image" src="https://github.com/user-attachments/assets/9035f726-1baa-4970-92a1-b2d6166547a0" />




