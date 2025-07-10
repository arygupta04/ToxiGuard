# Results (from ToxiGuard.ipynb)
~~| Label           | Loss  | Accuracy | Precision | Recall  |~~
~~|-----------------|-------|----------|-----------|---------|~~
~~| toxic           | 5.37% | 98.27%   | 98.19%    | 98.34%  |~~
~~| obscene         | 2.42% | 99.20%   | 99.24%    | 99.17%  |~~
~~| threat          | 0.84% | 99.76%   | 99.91%    | 99.61%  |~~
~~| insult          | 1.72% | 99.37%   | 99.44%    | 99.29%  |~~
~~| identity_hate   | 0.64% | 99.77%   | 99.74%    | 99.80%  |~~  


EDIT: after changing and retraing for a single model, the results on testing set is:
- **Test Loss:** 0.005
- **Test Precision:** 0.9779
- **Test Recall:** 0.9768
- **Test F1 score:** 0.9774

### Key Definitions

- **Precision**: The proportion of correctly identified positive cases out of all predicted positive cases. Higher precision ensures fewer false positives.
- **Recall**: The proportion of correctly identified positive cases out of all actual positive cases. Higher recall ensures fewer false negatives.
- **Accuracy**: The proportion of correctly classified messages (both positive and negative) out of all messages.

### Observations
- Very good results overall!

# How it works
## Training
1) **Preprocess the data:** First we need to have training data as 'X' and 'y', the comment and respective classifications for each label.
2) **Text Vectorisation:** This is a Natural Language Processing project, a sub-field of machine learning. Models cannot be trained in text format, thus Tensorflow's text vectorisation is used to convert text into numerical representations. Overall, a dictinary is built of each word corresponding to a integer, this is then used to convert input sentences into integer representation.
3) ~~**SMOTE:** Data given is imbalanced, the SMOTE tool adds data points to balance the data for each label. Thus, we have 5 datasets, different for each label.~~
   **Class weights:** SMOTE is not a good method for data generation in this case because it cannot directly generate more data. All it does is, find relevant and similar data points and tries to find points similar to it. So, we believe, SMOTE is not a good method to balance a textual data, and to solve this issue, we introduced class weights. The labels which have lesser occurances and if they are classified wrongly, the loss function is penalised more by these weights. This encourages the model to learn patterns associated with minority labels without artificially altering the dataset.
4) **Sequential Model:** ~~We have defined a binary classification model archutecture. As discussed, five models of this type will be made for each label. There are several layers, the first embedding layer converts each word into 32 dimentional vector.~~ The sequencial model has architecture: Embedding -> Bidirectional LSTM -> dense(128) -> dense(256) -> dense(128) -> dense(5, sigmoid). The bidirectional layer is important as it can process sentences in both directions and preserve its context. For example, the sentence "I don't hate you" is not toxic, but the association of the word "hate" in embedding layer is related to toxic, the word "don't" changes the context, which is what this layer takes care of. Then we have three dense or fully connected layers with Rectified Linear Unit (ReLU) function which learns features of words. This is followed by a final layer which gives probablitistic output of true or false classification for that label.
6) **Training and Testing:** 70:20:10 (Train:Val:Test) split. A callback is defined for saving the model after each epoch for each model. This is then tested on testing data split, and results are stored in results array.

---

## **Discord Bot**  

### **Overview**  
This project involves the development of a Discord bot designed to detect and manage toxic behavior in group chats. The bot uses machine learning models to classify messages and take appropriate actions to maintain a healthier online environment.  

### **Key Components**  

1. **toxicity_wrapper.py**  
   - Handles all prediction-related tasks for classifying the toxicity of messages.  
   - **Functionality:**  
     - Vectorizes the input text for processing.  
     - Passes the vectorized data to pre-trained machine learning models.  
     - The `predict` function aggregates predictions from all models.  
     - The `classify` function converts predictions into binary outcomes (1 or 0) based on predefined thresholds.  

2. **bot.py**  
   - Manages how the bot interacts with Discord servers and handles inappropriate messages.  
   - **Features:**  
     - Deletes flagged messages that violate community guidelines.  
     - Sends direct warnings to users, explaining why their message was flagged.  
     - Maintains seamless integration with Discord's API for real-time message monitoring and response.  

3. **ToxiGuard.ipynb**  
   - Implements the machine learning model used for toxicity detection.  
   - **Workflow:**  
     - Preprocesses and cleans the input dataset for training.  
     - Trains and evaluates models for various categories of toxic behavior.  
     - Stores the trained models, which are later used in `toxicity_wrapper.py`.  

---

# Pain Points and challenges
1) **Data imbalance:** We trained the model multiple times without realizing the dataset was imbalanced. As a result, we achieved high accuracy for the "toxic" label since it had the most data and was relatively balanced. However, the other labels, particularly "threat," struggled significantly. The precision for "threat" was 0, indicating the model failed to identify any true positives and classified everything as false. This occurred because only a small number of data points were labeled as "threat," and the model failed to learn meaningful patterns for this label due to the overwhelming majority of false classifications. We resolved this by class weights.
2) **Removing severe_toxic label:** This label was removed, as there was no clear definition or line as to what classifies as severe_toxic and toxic. Moreover, as shown in notebook, all the severe_toxic labels were also toxic. So, we decided to drop the severe_toxic column.
3) **Fixing data imbalance, training multiple models:** Earlier, we were training only one model to perform 5 classifications. We realised one model will not perform all classifications well, even after training for 7 epochs the accuracy remained 54%. This is perhaps because even if the model classifies one label wrong, the testing would make whole comment as false prediction. Instead what we realised we should test model for wach label. To solve this, we trained 5 seperate models, each trained to perform binary classification outputting weather a label is true or false. This gave us very high accuracy, precision, and recall, especially after applying SMOTE.
4) **Edit**: No longer using SMOTE, but class weights. Also Training one model only.
5) **Multilabel classification:** The problem was in designing the whole solution, one option was to train just one model, which did not work as described. Second option was to have classes and train one model but this meant 2^5 classes possiblity which is ridiculous. The problem is that each comment can be classified in multiple label, thus, training 5 seperate model was the best option.
