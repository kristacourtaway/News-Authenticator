# News Authenticator Python Code

Project Summary: News Authenticator using Natural Language Processing (NLP) and Machine Learning

Introduction:
This project focuses on developing and evaluating machine learning models to authenticate news using Natural Language Processing (NLP) techniques. The goal is to build models that can effectively differentiate between real and fake news articles based on their textual content. The code utilizes popular Python libraries like pandas, seaborn, scikit-learn, and pickle for data handling, visualization, and model implementation.

Data Import and Preprocessing:
The project begins by importing the required libraries and loading the dataset "fake_or_real_news.csv." The dataset contains labeled news articles categorized as either 'real' or 'fake.' After loading the data, its integrity is checked to ensure no missing values.

Exploratory Data Analysis (EDA):
The distribution of labels ('real' and 'fake') in the dataset is visualized using a count plot. The EDA allows an understanding of the class balance, which is crucial for evaluating model performance.

Train-Test Split:
To assess the model's generalization capability, the dataset is split into training and testing sets with a test size of 33%. The training set is used for model training, while the testing set evaluates the model's performance.

Text Preprocessing using NLP Techniques:
Text data preprocessing uses NLP techniques to convert the raw text into numerical features suitable for machine learning algorithms. Two popular techniques, CountVectorizer and TF-IDF (Term Frequency-Inverse Document Frequency), are applied to represent the text data.

CountVectorizer: 
It converts the text data into a matrix of word counts. Stop words (commonly occurring words like 'and', 'the', etc.) are removed during this process to reduce feature dimensionality.

TF-IDF: It calculates the importance of each word in a document relative to the entire dataset. Words frequent in a document but rare across the dataset are given higher weights.

Model Building and Evaluation:
Two machine learning models are trained and evaluated using the preprocessed text data:

Multinomial Naive Bayes (NB) Classifier:
 Two pipelines are constructed—one with CountVectorizer and the other with TF-IDF as feature extraction methods. The Multinomial NB classifier is used in both pipelines.

Passive Aggressive Classifier (PA Classifier): 
The TF-IDF features are fed into the Passive Aggressive Classifier for the news authenticator.

Performance Metrics:
The model's performance is evaluated using confusion matrices and accuracy scores. Confusion matrices help assess the true positive, true negative, false positive, and false negative predictions, while accuracy measures the proportion of correctly classified instances in the test set.

Model Deployment:
The best-performing model, the TF-IDF-based Passive Aggressive Classifier, is serialized and saved to a file named "final_model.pkl" using the pickle library. Serializing the model allows easy deployment for real-world use.

Conclusion:
The project demonstrates implementing NLP techniques and machine learning models for fake news detection. Using Python libraries, the code preprocesses textual data, builds machine learning pipelines, and evaluates the models' performance. The resulting TF-IDF-based Passive Aggressive Classifier showcases promising results in detecting fake news accurately. This work contributes to identifying misinformation in news articles, promoting reliable information dissemination, and combatting the spread of fake news.
