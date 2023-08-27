# News Authenticator Python Code

Project Summary: This project uses Natural Language Processing (NLP) techniques to create and assess machine learning models for news authentication. The goal is to distinguish between real and fake news articles based on their textual content. Python libraries like pandas, seaborn, sci-kit-learn, and Pickle are used for data management, visualization, and model implementation.

Introduction: The project aims to build and evaluate machine learning models for news authentication utilizing NLP techniques. The aim is to differentiate between real and fake news articles based on their textual content. The code employs prominent Python libraries such as pandas, seaborn, sci-kit-learn, and Pickle for data management, visualization, and model implementation tasks.

Data Import and Preprocessing: The project commences by importing necessary libraries and loading the "fake_or_real_news.csv" dataset. The dataset comprises labeled news articles categorized as either 'real' or 'fake.' After loading the data, a data integrity check is executed to ensure the absence of missing values.
Exploratory Data Analysis (EDA): Visualizing the distribution of labels ('real' and 'fake') in the dataset using a count plot facilitates understanding class balance—a crucial element for evaluating model performance.

Train-Test Split: The dataset undergoes division into training and testing sets, with a test size of 33%. The training set facilitates model training, while the testing set evaluates model performance.

Text Preprocessing using NLP Techniques: Text data preprocessing into numerical features suitable for machine learning involves NLP techniques. The text data is represented using two prevalent methods: CountVectorizer and TF-IDF (Term Frequency-Inverse Document Frequency).
CountVectorizer: This method transforms text data into a matrix of word counts. During this process, common words such as 'and' and 'the' are eliminated to reduce feature dimensionality.

TF-IDF: TF-IDF computes the importance of words in a document relative to the entire dataset. Words frequently present in a document but infrequent across the dataset receive higher weights.

Model Building and Evaluation: The code trains and assesses two machine learning models using preprocessed text data:
Multinomial Naive Bayes (NB) Classifier: The code constructs two pipelines with CountVectorizer and TF-IDF as feature extraction methods. The Multinomial NB classifier is utilized in both pipelines.

Passive Aggressive Classifier (PA Classifier): The Passive Aggressive Classifier, employing TF-IDF features, is used for news authentication.
Performance Metrics: Model performance is gauged through confusion matrices and accuracy scores. Confusion matrices aid in evaluating true positive, true negative, false positive, and false negative predictions. Accuracy measures the proportion of correctly classified instances in the test set.
Model Deployment: The best-performing model, the TF-IDF-based Passive Aggressive Classifier, is serialized and saved as "final_model.pkl" using the pickle library. Serialization facilitates smooth deployment in real-world scenarios.

Conclusion: This project showcases the implementation of NLP techniques and machine learning models for detecting fake news. Python libraries are utilized to preprocess text data, construct machine learning pipelines, and evaluate model performance. The resulting TF-IDF-based passive-aggressive classifier exhibits promising accuracy in identifying fake news. The project contributes to the fight against misinformation, promotes credible information dissemination, and mitigates the proliferation of false news.
