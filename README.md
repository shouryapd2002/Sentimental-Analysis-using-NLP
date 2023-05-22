# Sentimental-Analysis-using-NLP
a guide to sentimental analysis using NLP

ABSTRACT

Sentiment analysis or opinion mining is one of the major tasks of NLP (Natural Language Processing). Sentiment analysis has gained much attention in recent years. In this paper, we aim to tackle the problem of sentiment polarity categorization, which is one of the fundamental problems of sentiment analysis. A general process for sentiment polarity categorization is proposed with detailed process descriptions. Data used in this study are online product reviews collected from Amazon.com.</pre>
Experiments for both sentence-level categorization and review-level categorization are performed with promising outcomes. At last, we also give insight into our future work on sentiment analysis.

INTRODUCTION

Sentiment Analysis, as the name suggests, means to identify the view or emotion behind a situation. It basically means to analyze and find the emotion or intent behind a piece of text or speech or any mode of communication.
In this article, we will focus on the sentiment analysis of text data.


We, humans, communicate with each other in a variety of languages, and any language is just a mediator or a way in which we try to express ourselves. And, whatever we say has a sentiment associated with it. It might be positive or negative or it might be neutral as well.


REQUIREMENT ANALYSIS


Suppose, there is a fast-food chain company and they sell a variety of different food items like burgers, pizza, sandwiches, milkshakes, etc. They have created a website to sell their food and now the customers can order any food item from their website and they can provide reviews as well, like whether they liked the food or hated it.
* User Review 1: I love this cheese sandwich, it’s so delicious.
* User Review 2: This chicken burger has a very bad taste.
* User Review 3: I ordered this pizza today.
So, as we can see that out of these above 3 reviews,










The first review is definitely a positive one and it signifies that the customer was really happy with the sandwich.
The second review is negative, and hence the company needs to look into their burger department.
And, the third one doesn’t signify whether that customer is happy or not, and hence we can consider this as a neutral statement.
By looking at the above reviews, the company can now conclude that it needs to focus more on the production and promotion of their sandwiches as well as improve the quality of their burgers if they want to increase their overall sales.
But, now a problem arises, that there will be hundreds and thousands of user reviews for their products and after a point of time it will become nearly impossible to scan through each user review and come to a conclusion.
Neither can they just come up with a conclusion by taking just 100 reviews or so, because maybe the first 100-200 customers were having similar taste and liked the sandwiches, but over time when the no. of reviews increases, there might be a situation where the positive reviews are overtaken by more no. of negative reviews.
Therefore, this is where the Sentiment Analysis Model comes into play, which takes in a huge corpus of data having user reviews and finds a pattern and comes up with a conclusion based on real evidence rather than assumptions made on a small sample of data.
(We will explore the working of a basic Sentiment Analysis model later in this article.)
We can even break these principal sentiments(positive and negative) into smaller sub sentiments such as “Happy”, “Love”, ”Surprise”, “Sad”, “Fear”, “Angry” etc. as per the needs or business requirement.
REAL-WORLD EXAMPLE


* There was a time when the social media
services like Facebook used to just have two emotions associated with each post, i.e You can like a post or you can leave the post without any reaction and that basically signifies that you didn’t like it.
* But, over time these reactions to post have
changed and grew into more granular sentiments which we see as of now, such as “like”, “love”, “sad”,
“angry” etc.


  

And, because of this upgrade, when any company promotes their products on Facebook, they receive more specific reviews which will help them to enhance the customer experience.
And because of that, they now have more granular control on how to handle their consumers, i.e. they can target the customers who are just “sad” in a different way as compared to customers who are “angry”, and come up with a business plan accordingly because nowadays, just doing the bare minimum is not enough.
Now, as we said we will be creating a Sentiment Analysis Model, but it’s easier said than done.
As we humans communicate with each other in a way that we call Natural Language which is easy for us to interpret but it’s much more complicated and messy if we really look into it.
Because there are billions of people and they have their own style of communicating, i.e. a lot of tiny variations are added to the language and a lot of sentiments are attached to it which is easy for us to interpret but it becomes a challenge for the machines.
This is why we need a process that makes the computers understand the Natural Language as we humans do, and this is what we call Natural Language Processing(NLP). And, as we know Sentiment Analysis is a sub-field of NLP and with the help of machine learning techniques, it tries to identify and extract the insights.
IMPLEMENTATION




<pre>#importing all the Librarires required</pre>
<pre>!pip3 install seaborn</pre>
<pre>import pandas as pd</pre>
<pre>import matplotlib.pyplot as plt</pre>
<pre>import seaborn as sns</pre>
<pre>from wordcloud import WordCloud</pre>
<pre>import re</pre>
Requirement already satisfied: seaborn in c:\users\shourya\anaconda3\lib\site-packages (0.12.2)
Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in c:\users\shourya\anaconda3\lib\site-packages (from seaborn) (3.7.0)
Requirement already satisfied: pandas>=0.25 in c:\users\shourya\anaconda3\lib\site-packages (from seaborn) (1.5.3)
Requirement already satisfied: numpy!=1.24.0,>=1.17 in c:\users\shourya\anaconda3\lib\site-packages (from seaborn) (1.23.5)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (4.25.0)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.8.2)
Requirement already satisfied: pillow>=6.2.0 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (9.4.0)
Requirement already satisfied: cycler>=0.10 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (0.11.0)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.4.4)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (3.0.9)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.0.5)
Requirement already satisfied: packaging>=20.0 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (22.0)
Requirement already satisfied: pytz>=2020.1 in c:\users\shourya\anaconda3\lib\site-packages (from pandas>=0.25->seaborn) (2022.7)
Requirement already satisfied: six>=1.5 in c:\users\shourya\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn) (1.16.0)
<pre>#installing wordcloud</pre>
<pre>!pip3 install wordcloud</pre>
Requirement already satisfied: wordcloud in c:\users\shourya\anaconda3\lib\site-packages (1.9.2)
Requirement already satisfied: matplotlib in c:\users\shourya\anaconda3\lib\site-packages (from wordcloud) (3.7.0)
Requirement already satisfied: pillow in c:\users\shourya\anaconda3\lib\site-packages (from wordcloud) (9.4.0)
Requirement already satisfied: numpy>=1.6.1 in c:\users\shourya\anaconda3\lib\site-packages (from wordcloud) (1.23.5)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.0.5)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib->wordcloud) (3.0.9)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.4.4)
Requirement already satisfied: cycler>=0.10 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.11.0)
Requirement already satisfied: packaging>=20.0 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib->wordcloud) (22.0)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib->wordcloud) (4.25.0)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\shourya\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.2)
Requirement already satisfied: six>=1.5 in c:\users\shourya\anaconda3\lib\site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
<pre>#nltk – Natural Language Toolkit is a collection of libraries for natural language processing
<pre>#stopwords – a collection of words that don’t provide any meaning to a sentence</pre>
<pre>#WordNetLemmatizer – used to convert different forms of words into a single item but still keeping the</pre>
<pre>import nltk</pre>
<pre>nltk.download('stopwords')</pre>
<pre>nltk.download('wordnet')</pre>
<pre>nltk.download('omw-1.4')</pre>
<pre>from nltk.corpus import stopwords</pre>
<pre>from nltk.stem import WordNetLemmatizer</pre>
[nltk_data] Downloading package stopwords to</pre>
[nltk_data]     C:\Users\Shourya\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\Shourya\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package omw-1.4 to
[nltk_data]     C:\Users\Shourya\AppData\Roaming\nltk_data...
[nltk_data]   Package omw-1.4 is already up-to-date!
<pre>#Scikit-Learn (Machine Learning Library for Python)</pre>
<pre>#CountVectorizer – transform text to vectors</pre>
<pre>#RandomizedSearchCV – for hyperparameter tuning</pre>
<pre>#RandomForestClassifier – machine learning algorithm for classification</pre>
<pre>#Accuracy Score – no. of correctly classified instances/total no. of instances</pre>
<pre>#Precision Score – the ratio of correctly predicted instances over total positive instances</pre>
<pre>#Recall Score – the ratio of correctly predicted instances over total instances in that class</pre>
<pre>#Roc Curve – a plot of true positive rate against false positive rate</pre>
<pre>#Classification Report – report of precision, recall and f1 score</pre>
<pre>#Confusion Matrix – a table used to describe the classification models</pre>
<pre>!pip install -U scikit-learn</pre>
<pre>from sklearn.model_selection import GridSearchCV</pre>
<pre>from sklearn.ensemble import RandomForestClassifier</pre>
<pre>from sklearn.metrics import roc_curve</pre>
<pre>from sklearn.metrics import auc</pre>
<pre>from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, classification_report</pre>
Requirement already satisfied: scikit-learn in c:\users\shourya\anaconda3\lib\site-packages (1.2.2)
Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\shourya\anaconda3\lib\site-packages (from scikit-learn) (2.2.0)
Requirement already satisfied: joblib>=1.1.1 in c:\users\shourya\anaconda3\lib\site-packages (from scikit-learn) (1.1.1)
Requirement already satisfied: numpy>=1.17.3 in c:\users\shourya\anaconda3\lib\site-packages (from scikit-learn) (1.23.5)
Requirement already satisfied: scipy>=1.3.2 in c:\users\shourya\anaconda3\lib\site-packages (from scikit-learn) (1.10.0)
<pre><pre>#emotion Dataset includes train.txt,test.txt,val.txt
<pre>df_train = pd.read_csv("train.txt",delimiter=';',names=['text','label'])
<pre>df_val = pd.read_csv("val.txt",delimiter=';',names=['text','label'])
<pre>df = pd.concat([df_train,df_val])
<pre>df.reset_index(inplace=True,drop=True)
<pre>#Now, we will create a custom encoder to convert categorical target labels to numerical form, i.e. (0 and 1)
<pre>def custom_encoder(df):
    df.replace(to_replace ="surprise", value =1, inplace=True)
    df.replace(to_replace ="love", value =1, inplace=True)
    df.replace(to_replace ="joy", value =1, inplace=True)
    df.replace(to_replace ="fear", value =0, inplace=True)
    df.replace(to_replace ="anger", value =0, inplace=True)
    df.replace(to_replace ="sadness", value =0, inplace=True)
<pre>custom_encoder(df['label'])
<pre>#Data Pre-Processing
<pre>#object of WordNetLemmatizer
<pre>lm = WordNetLemmatizer()
<pre>def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
<pre>    return corpus
<pre>corpus = text_transformation(df['text'])
<pre>#Bag Of Words(BOW)
<pre>from pylab import rcParams
<pre>rcParams['figure.figsize'] = 20,8
<pre>word_cloud = ""
<pre>for row in corpus:
    for word in row:
        word_cloud+=" ".join(word)
<pre>wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
<pre>plt.imshow(wordcloud)
<matplotlib.image.AxesImage at 0x20d88e5a200>
  


#Estimator or model – RandomForestClassifier in our case
#parameters – dictionary of hyperparameter names and their values
#cv – signifies cross-validation folds
#return_train_score – returns the training scores of the various models
# n_jobs – no. of jobs to run parallelly (“-1” signifies that all CPU cores will be used which reduces the training time drastically)
#First, We will create a dictionary, “parameters” which will contain the values of different hyperparameters.
#We will pass this as a parameter to RandomSearchCV to train our random forest classifier model using all possible 
#combinations of these parameters to find the best model.
<pre>from sklearn.feature_extraction.text import CountVectorizer
<pre>cv = CountVectorizer(ngram_range=(1,2))
<pre>traindata = cv.fit_transform(corpus)
<pre>X = traindata
<pre>y = df.label
<pre>parameters = {'max_features': ('auto','sqrt'),
             'n_estimators': [50, 10, 15],
             'max_depth': [2, 5, None],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 5, 10],
             'bootstrap': [True, False]}
<pre>from sklearn.model_selection import RandomizedSearchCV
<pre>rf_RandomGrid = RandomizedSearchCV(RandomForestClassifier(),parameters,n_iter=10,cv=5,return_train_score=True,n_jobs=-1)
<pre>rf_RandomGrid.fit(X,y)
<pre>rf_RandomGrid.best_params_
{'n_estimators': 15,
 'min_samples_split': 10,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': None,
 'bootstrap': True}
<pre>#And then, we can view all the models and their respective parameters,
<pre>#mean test score and rank as  GridSearchCV stores all the results in the cv_results_ attribute.
<pre>for i in range(10):
    print('Parameters: ',rf_RandomGrid.cv_results_['params'][i])
    print('Mean Test Score: ',rf_RandomGrid.cv_results_['mean_test_score'][i])
    print('Rank: ',rf_RandomGrid.cv_results_['rank_test_score'][i])
Parameters:  {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'auto', 'max_depth': None, 'bootstrap': False}
Mean Test Score:  0.9354444444444443
Rank:  3
Parameters:  {'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_features': 'auto', 'max_depth': None, 'bootstrap': True}
Mean Test Score:  0.8012777777777778
Rank:  4
Parameters:  {'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 2, 'bootstrap': False}
Mean Test Score:  0.5443888888888889
Rank:  9
Parameters:  {'n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': 'auto', 'max_depth': 5, 'bootstrap': False}
Mean Test Score:  0.5451111111111111
Rank:  6
Parameters:  {'n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 'auto', 'max_depth': 5, 'bootstrap': False}
Mean Test Score:  0.5449999999999999
Rank:  7
Parameters:  {'n_estimators': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': None, 'bootstrap': True}
Mean Test Score:  0.9411111111111111
Rank:  2
Parameters:  {'n_estimators': 15, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True}
Mean Test Score:  0.9425555555555555
Rank:  1
Parameters:  {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 5, 'bootstrap': False}
Mean Test Score:  0.5895
Rank:  5
Parameters:  {'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 5, 'bootstrap': False}
Mean Test Score:  0.5446666666666666
Rank:  8
Parameters:  {'n_estimators': 50, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'max_depth': 2, 'bootstrap': True}
Mean Test Score:  0.5443888888888889
Rank:  9
<pre>#Now, we will choose the best parameters obtained from GridSearchCV and create a final random forest classifier model and then train our new model.
<pre>rfc = RandomForestClassifier(max_features=rf_RandomGrid.best_params_['max_features'],
                                      max_depth=rf_RandomGrid.best_params_['max_depth'],
                                      n_estimators=rf_RandomGrid.best_params_['n_estimators'],
                                      min_samples_split=rf_RandomGrid.best_params_['min_samples_split'],
                                      min_samples_leaf=rf_RandomGrid.best_params_['min_samples_leaf'],
                                      bootstrap=rf_RandomGrid.best_params_['bootstrap'])
<pre>rfc.fit(X,y)
<pre>RandomForestClassifier(min_samples_split=10, n_estimators=15)
<pre>#Test Data Transformation
<pre>test_df = pd.read_csv('test.txt',delimiter=';',names=['text','label'])
<pre>X_test,y_test = test_df.text,test_df.label
<pre>#encode the labels into two classes , 0 and 1
<pre>test_df = custom_encoder(y_test)
<pre>#pre-processing of text
<pre>test_corpus = text_transformation(X_test)
<pre>#convert text data into vectors
<pre>testdata = cv.transform(test_corpus)
<pre>#predict the target
<pre>predictions = rfc.predict(testdata)
<pre>rcParams['figure.figsize'] = 10,5
<pre>cm=ConfusionMatrixDisplay.__init__(y_test,predictions)
<pre>acc_score = accuracy_score(y_test,predictions)
<pre>pre_score = precision_score(y_test,predictions)
<pre>rec_score = recall_score(y_test,predictions)
<pre>print('Accuracy_score: ',acc_score)
<pre>print('Precision_score: ',pre_score)
<pre>print('Recall_score: ',rec_score)
<pre>print("-"*50)
<pre>cr = classification_report(y_test,predictiprint(cr)
Accuracy_score:  0.9475
Precision_score:  0.957351290684624
Recall_score:  0.9271739130434783
--------------------------------------------------
              precision    recall  f1-score   support

           0       0.94      0.96      0.95      1080
           1       0.96      0.93      0.94       920

    accuracy                           0.95      2000
   macro avg       0.95      0.95      0.95      2000
weighted avg       0.95      0.95      0.95      2000

<pre>#Model Evaluation
<pre>predictions_probability = rfc.predict_proba(testdata)
<pre>fpr,tpr,thresholds = roc_curve(y_test,predictions_probability[:,1])
<pre>plt.plot(fpr,tpr)
<pre>plt.plot([0,1])
<pre>plt.title('ROC Curve')
<pre>plt.xlabel('False Positive Rate')
<pre>plt.ylabel('True Positive Rate')
<pre>plt.show()
  


<pre>#Model Prediction
<pre>def expression_check(prediction_input):
    if prediction_input == 0:
        print("Input statement has Negative Sentiment.")
    elif prediction_input == 1:
        print("Input statement has Positive Sentiment.")
    else:
        print("Invalid Statement.")
<pre># function to take the input statement and perform the same transformations we did earlier
<pre>def sentiment_predictor(input):
    input = text_transformation(input)
    transformed_input = cv.transform(input)
    prediction = rfc.predict(transformed_input)
    expression_check(prediction)
<pre>input1 = ["i feel more virtuous than when i eat veggies dipped in hummus."]
<pre>input2 = ["I bought a new phone and it's so bad."]
<pre>sentiment_predictor(input1)
<pre>sentiment_predictor(input2)
<pre>Input statement has Positive Sentiment.
<pre>Input statement has Negative Sentiment.
________________


  

RESULT






* We have successfully created a Sentimental Analysis model using NLP.
* We Revised New Libraries for optimal runtime and ensured letter runtime.
* We use Random Forest Classifier and RandomizedSearch (i.e RandomizedSearchCV instead of GridSearchCV)


REFERENCES


* We have used the dataset that contains 3 separate files named train.txt, test.txt and val.txt.
* https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/
