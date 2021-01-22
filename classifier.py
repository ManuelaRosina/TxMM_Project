from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import nltk
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import string
import numpy as np
import pandas as pd
from frog import Frog, FrogOptions
import re
import matplotlib.pyplot as plt
import matplotlib.dates
import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

# label 0 = not funny
# label 1 = funny
# label 3 = ambigous
# none/2 = I don't know/understand

def load_data():
    data = pd.read_csv('Data/train_data_cleaned_labeled.csv')
    data_unlabeled = pd.read_csv('Data/test_data_cleaned.csv')
    return data, data_unlabeled

def punctuation(message, bag_of_words):
    features = []
    heavy_punctuation = re.compile('([!?.]{2,})')
    tmp = heavy_punctuation.findall(message)
    # test if heavy punctuation is used
    if tmp != []:
        features.append(1)
    else:
        features.append(0)

    # number of sentences with question marks
    question = re.compile('(\w+[?])')
    tmp = question.findall(message)
    features.append(len(tmp))

    # number of sentences with exclamation marks
    question = re.compile('(\w+[!])')
    tmp = question.findall(message)
    features.append(len(tmp))

    # number of normal sentences
    question = re.compile('(\w+[.])')
    tmp = question.findall(message)
    features.append(len(tmp))

    return features

def hashtags(message, bag_of_words):
    # number of hashtags
    hashtag = re.compile('(#\w+)')
    tmp = hashtag.findall(message)
    return len(tmp)

def quote(message, bag_of_words):
    # search if quotes are used
    quotes = re.compile('(\".+\"|\'.+\')')
    tmp = quotes.findall(message)
    return len(tmp)

def negation(message, bag_of_words):
    # negations
    negations = re.compile('(niet)|(Niet)')
    tmp = negations.findall(message)
    return len(tmp)

def adult(message, bag_of_words):
    features = []
    # adult slang
    adult_words = ['neuken', 'naaien', 'sex', 'seks', 'porno', 'vibrator']
    for word in adult_words:
        features.append(sum([1 for x in bag_of_words if x.lower() == word]))

    return features

def human(message, bag_of_words):
    features = []
    # Human words ---> reduces precision
    human = ['ik', 'je', 'jij', 'hij', 'zij', 'meisje', 'jonge', 'iemand']
    for word in human:
        features.append(sum([1 for x in bag_of_words if x.lower() == word]))

    return features

def animal(message, bag_of_words):
    # animal names
    pass

def repetition(message, bag_of_words):
    features = []
    # repetition ---> reduces precision
    long_words = [x for x in bag_of_words if len(x) > 4]
    unique_long_words = set(long_words)
    features.append(len(long_words) - len(unique_long_words))

    # alliteration
    first_letter = [x[0] for x in bag_of_words if x[0].isalpha()]
    first_letter = ''.join(first_letter)
    matcher = re.compile(r'(.)\1*')
    grouped = [match.group() for match in matcher.finditer(first_letter)]
    alliteration = [x for x in grouped if len(x) > 1]
    features.append(len(alliteration))

    return features

def neg_orient(message, bag_of_words):
    features = []
    # negative orientation ---> reduces nearly all measures
    negative = ['slecht', 'slechts', 'niemand', 'nee', 'verslaving', 'jaloezie', 'jaloers', 'missen', 'mist']
    for word in negative:
        features.append(sum([1 for x in bag_of_words if x.lower() == word]))

    return features

def brackets(message, bag_of_words):
    features = []
    # bracets
    brackets = ['(', ')', '[', ']', '{', '}']
    for b in brackets:
        features.append(sum([1 for x in bag_of_words if b in x]))

    return features

def other_metrics(message, bag_of_words):
    features = []
    nr_of_words = len(bag_of_words)
    features.append(nr_of_words)

    # number of alphabetical and nonalphabetical characters
    features.append(sum([1 for x in message if x.isalpha()]))
    features.append(sum([1 for x in message if not x.isalpha()]))

    # count all uppercase words and words starting with an uppercase letter
    features.append(sum([1 for x in bag_of_words if x.isupper()]))
    features.append(sum([1 for x in bag_of_words if x[0].isupper()]))

    return features

def extract_features(message, leave_out=0):
    #print(message)
    bag_of_words = [x for x in wordpunct_tokenize(message)]

    features = []

    if leave_out == 0:
        features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 1:
        #features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 2:
        features.extend(punctuation(message, bag_of_words))
        #features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 3:
        features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        #features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 4:
        features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        #features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 5:
        features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        #features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 6:
        features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        #features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 7:
        features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        #features.extend(neg_orient(message, bag_of_words))
        features.extend(other_metrics(message, bag_of_words))
    elif leave_out == 8:
        features.extend(punctuation(message, bag_of_words))
        features.append(hashtags(message, bag_of_words))
        features.append(quote(message, bag_of_words))
        #features.extend(brackets(message, bag_of_words))
        features.append(negation(message, bag_of_words))
        features.extend(adult(message, bag_of_words))
        #features.extend(human(message, bag_of_words))
        features.extend(repetition(message, bag_of_words))
        features.extend(neg_orient(message, bag_of_words))
        #features.extend(other_metrics(message, bag_of_words))

    #print(len(features))
    return features

def classify(train_features, train_labels, test_features, probability=False):
    clf = SVC(kernel='linear', decision_function_shape='ovo', probability=probability)
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return pred
    #roc(vote_count, pred, max(vote_count))

    #print()
    #print("popularity based label")
    #evaluate(popular_labels, pred, print_out=True)

    """
    print()
    print("K-Means")
    kmeans = KMeans(n_clusters=2, random_state=0).fit(train_features)
    pred = kmeans.predict(test_features)
    print("hand label")
    evaluate(test_labels, pred, print_out=True)
    print("popularity")
    evaluate(popular_labels, pred, print_out=True)
    """

    #clustering = AffinityPropagation().fit(train_features)
    #return clustering.predict(test_features)

# Evaluate predictions (y_pred) given the ground truth (y_true)
def evaluate(y_true, y_pred, print_out=False):
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    tnr = tn/(tn+fp)

    if print_out:
        print("Accuracy: {}".format(accuracy))
        print("Precision: %f" % precision)
        print("Recall (TPR): %f" % recall)
        print("F1-score: %f" % f1_score)
        print("true negative rate: {}".format(tnr))

    return accuracy, precision, recall, f1_score, tnr

def roc(vote_count, pred, true_label, max_threshold):
    tpr, fpr = [], []
    print("max threshold: ", max_threshold)
    for threshold in range(max_threshold):
        popular_label = [1 if i > threshold else 0 for i in vote_count]
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true_label, popular_label).ravel()

        tpr.append(tp / (tp + fn))
        # fpr = fp / (fp + tn)
        spec = tn / (tn + fp)
        fpr.append(1 - spec)

    print("TPR: ", tpr)
    print("FPR: ", fpr)
    plt.figure()
    lw = 2
    plt.plot(range(max_threshold), tpr, color='darkorange', lw=lw, label="True Positive Rate")
    plt.plot(range(max_threshold), fpr, lw=lw, label="False Positive Rate")
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.show()
    plt.title('FPR and TPR over various threshold values')

    plt.plot(fpr, tpr)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title('FPR and TPR over various threshold values')
    #plt.legend()
    plt.show()

def plot_feature_groups(validation, test, label="", title=""):
    feature_labels = ["All", "Punctuation", "Hashtag", "Quotes", "Negation", "Adult language",
                       "Repitition", "Negativ orientation", "Others"]

    x = np.arange(len(feature_labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, validation, width, label='Validation')
    plt.axhline(validation[0], color='r', linestyle='--')
    rects2 = ax.bar(x + width/2, test, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

def diff_labels(data, label_hand, label_pop):
    data['same'] = [i == j for i, j in zip(label_hand, label_pop)]
    same_data = data[data['same'] == True]
    diff_data = data[data['same'] == False]
    dates_same = [x.time() for x in same_data['collection Time']]
    x_dt_same = [datetime.datetime.combine(datetime.date.today(), t) for t in dates_same]
    dates_diff = [x.time() for x in diff_data['collection Time']]
    x_dt_diff = [datetime.datetime.combine(datetime.date.today(), t) for t in dates_diff]

    fig, ax = plt.subplots()
    ax.scatter(x_dt_same, same_data['vote_count'], label='same label')
    ax.scatter(x_dt_diff, diff_data['vote_count'], label='different label')
    plt.axhline(40, color='r', linestyle='--')
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    plt.title("Posts that are labeled same and different")
    plt.xlabel("Post time")
    plt.ylabel("Nr. of votes")
    plt.legend()
    plt.show()

data, data_unlabeled = load_data()
data['collection Time'] = pd.to_datetime(data['collection Time'])
data['created_at'] = pd.to_datetime(data['created_at'])
threshold = 40
data['popular'] = data['vote_count'] > threshold
data['post_id'] = data['post_id'].astype(str)
data['humour'] = data['humour'].fillna(2)
data = data.dropna()

data_unlabeled['collection Time'] = pd.to_datetime(data_unlabeled['collection Time'])
data_unlabeled['created_at'] = pd.to_datetime(data_unlabeled['created_at'])
data_unlabeled['popular'] = data_unlabeled['vote_count'] > threshold
data_unlabeled['post_id'] = data_unlabeled['post_id'].astype(str)
data_unlabeled = data_unlabeled.dropna()

train_data = data[0:1000].copy()
X_train, X_validation = train_test_split(train_data, test_size=0.33, random_state=3)
X_train, X_validation = X_train.copy(), X_validation.copy()

test_data = data[1001:].copy()

train_messages = X_train['message']
counts = X_train.groupby(['humour', 'popular']).count()[['message']]

fig, ax = plt.subplots()
counts.unstack(level=0).plot(kind='bar', ax=ax)
ax.legend(["no-humorous", "humorous", "unknown", "ambiguous"])
plt.title("correlation in train data using a threshold of {}".format(threshold))
ax.set_xticklabels(['non-popular', 'popular'], rotation=0)
plt.show()

# count 2 and 3 to non-humours
X_train['humour'] = X_train['humour'].replace([2,3], 0)
train_label_true = X_train['humour']
train_label_popular = X_train['popular']

X_validation['humour'] = X_validation['humour'].replace([2,3], 0)
val_label_true = X_validation['humour']
val_label_popular = X_validation['popular']

test_data['humour'] = test_data['humour'].replace([2,3], 0)
test_label_true = test_data['humour']
test_label_popular = test_data['popular']

unlabeled_popular = data_unlabeled['popular']

diff_labels(X_validation, val_label_true, val_label_popular)
diff_labels(data, data['humour'], data['popular'])
diff_labels(test_data, test_label_true, test_label_popular)

#counts = data.groupby(['humour', 'popular']).count()[['message']]
#counts.unstack(level=0).plot(kind='bar', subplots=False)
#plt.show()

# plot of creation time and vote count

same_data = X_train[X_train['humour'] == 0]
diff_data = X_train[X_train['humour'] == 1]
dates_same = [x.time() for x in same_data['collection Time']]
x_dt_same = [datetime.datetime.combine(datetime.date.today(), t) for t in dates_same]
dates_diff = [x.time() for x in diff_data['collection Time']]
x_dt_diff = [datetime.datetime.combine(datetime.date.today(), t) for t in dates_diff]

#creation_times = X_train['created_at']
#last_upvote = X_train['vote_count']
#dates = [x.time() for x in X_train['collection Time']]
#x_dt = [datetime.datetime.combine(datetime.date.today(), t) for t in dates]

fig, ax = plt.subplots()
ax.scatter(x_dt_same, same_data['vote_count'], label='humorous')
ax.scatter(x_dt_diff, diff_data['vote_count'], label='non-humorous')
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
fig.autofmt_xdate()
plt.title("Post time vs. number of votes of train data")
plt.xlabel("Post time")
plt.ylabel("Nr. of votes")
plt.legend()
plt.show()

train_total_accuracy = []
train_total_precision = []
train_total_recall = []
train_total_f1_score = []
train_total_tnr = []

test_total_accuracy = []
test_total_precision = []
test_total_recall = []
test_total_f1_score = []
test_total_tnr = []

thresh_total_accuracy = []
thresh_total_precision = []
thresh_total_recall = []
thresh_total_f1_score = []
thresh_total_tnr = []

feature_labels = ["All", "Punctuation", "Hashtag", "Quotes", "Negation", "Adult language",
                       "Repitition", "Negativ orientation", "Others"]

for i in range(1):
    print("Left out group {}".format(feature_labels[i]))
    train_features = list(map(lambda x: extract_features(x, leave_out=i), list(X_train['message'])))
    val_features = list(map(lambda x: extract_features(x, leave_out=i), list(X_validation['message'])))
    test_features = list(map(lambda x: extract_features(x, leave_out=i), list(test_data['message'])))
    unlabeled_features = list(map(lambda x: extract_features(x, leave_out=i), list(data_unlabeled['message'])))

    pred = classify(train_features, train_label_true, val_features)
    val_label_popular = X_validation["vote_count"] >= 40
    print("Validation:")
    accuracy, precision, recall, f1_score, tnr = evaluate(val_label_true, pred, print_out=True)
    print()
    #accuracy, precision, recall, f1_score, tnr = evaluate(val_label_popular, pred, print_out=True)
    #print()
    train_total_accuracy.append(accuracy)
    train_total_precision.append(precision)
    train_total_recall.append(recall)
    train_total_f1_score.append(f1_score)
    train_total_tnr.append(tnr)
    """for thresh in [10,15,20,25,30,35,40,45,50]:
        #print("threshold: ", thresh)
        val_label_popular = X_validation["vote_count"] >= thresh
        accuracy, precision, recall, f1_score, tnr = evaluate(val_label_popular,pred, print_out=False)
        thresh_total_accuracy.append(accuracy)
        thresh_total_precision.append(precision)
        thresh_total_recall.append(recall)
        thresh_total_f1_score.append(f1_score)
        thresh_total_tnr.append(tnr)
    thresh = [10,15,20,25,30,35,40,45,50]
    plt.plot(thresh, thresh_total_accuracy, '.-', label='Accuracy')
    plt.plot(thresh, thresh_total_precision, '.-', label='Precision')
    plt.plot(thresh, thresh_total_recall, '.-', label='Recall')
    plt.plot(thresh, thresh_total_f1_score, '.-', label='F1')
    plt.plot(thresh, thresh_total_tnr, '.-', label='TNR')
    plt.xlabel('Threshold')
    plt.legend()
    plt.title("Metrics with different threshold")
    plt.show()

    val_label_popular = X_validation["vote_count"] >= 40
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(val_label_true, val_label_popular).ravel()
    print("TP: ", tp)
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)
    print()

    test_label_popular = test_data["vote_count"] >= 40
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(test_label_true, test_label_popular).ravel()
    print("TP: ", tp)
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)"""



    #roc(X_validation["vote_count"], pred, val_label_true, max(X_validation["vote_count"]))

    print("Test")
    pred = classify(train_features, train_label_true, test_features)
    accuracy, precision, recall, f1_score, tnr = evaluate(test_label_true, pred, print_out=True)
    print()
    #test_label_popular = test_data["vote_count"] >= 40
    #accuracy, precision, recall, f1_score, tnr = evaluate(test_label_popular, pred, print_out=True)
    test_total_accuracy.append(accuracy)
    test_total_precision.append(precision)
    test_total_recall.append(recall)
    test_total_f1_score.append(f1_score)
    test_total_tnr.append(tnr)
    print("Unlabeled")
    pred = classify(train_features, train_label_true, unlabeled_features)
    accuracy, precision, recall, f1_score, tnr = evaluate(unlabeled_popular, pred, print_out=True)
    print()

#plot_feature_groups(train_total_accuracy, test_total_accuracy, label="Accuracy")
#plot_feature_groups(train_total_precision, test_total_precision, label="Precision")
#plot_feature_groups(train_total_recall, test_total_recall, label="Recall")
#plot_feature_groups(train_total_f1_score, test_total_f1_score, label="F1")
#plot_feature_groups(train_total_tnr, test_total_tnr, label="TNR")
#evaluate(test_labels, pred, print_out=True)
#print()
#evaluate(popular_test_labels, pred, print_out=True)


"""np_df = data.as_matrix()
print(np_df[0][0])
frog = Frog(FrogOptions(parser=False))


output = frog.process(np_df[0][0])
print("PARSED OUTPUT=",output)"""