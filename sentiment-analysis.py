from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

vectorizer = CountVectorizer(binary = 'true')

# Reading data
def get_data(path):
    with open(path, 'r') as sentence:
        data = sentence.read().split('\n')
    return data

# Adding data in list
def adding_data_list(data):
    separate_data_sentence_and_type = []
    for line in data:
        separate_data_sentence_and_type.append(line.split('\t'))
    return separate_data_sentence_and_type

# Separate data of training and validate
def separate_data(data):
    total = len(data)
    training = []
    validate = []
    total_training = 0.75 * total
    total_training = int(total_training)
    for i in range(0, total):
        if i < total_training:
            training.append(data[i])
        else:
            validate.append(data[i])

    return training, validate

# Training data
def training(data, vectorizer):
    comments = []
    answers = []
    total = len(data)
    for i in range(0, total):
        if len(data[i]) == 2: # verify exists lists emptys
            comments.append(data[i][0])
            answers.append(data[i][1])

    comments = vectorizer.fit_transform(comments)

    return BernoulliNB().fit(comments, answers)

def analyze_sentence(classificator, vectorizer, sentence):
    return sentence, classificator.predict(vectorizer.transform([sentence]))

def analyze_data_validate(data):
    success = 0
    fail = 0
    total = len(data)
    for i in range(0, total):
        analyze = analyze_sentence(classificator, vectorizer, data[i][0])
        if len(data[i]) == 2 and analyze[1][0] == data[i][1]: # verify exists lists emptys and validate classificator
            success += 1
        else:
            fail += 1
    print('Total de amostras de validação: %d' % total)
    print('Acertos: %d' % success)
    print('Erros: %d' % fail)
    print('Percentual de acertos: %.2f%%' % (success/total*100))
    print('Percentual de erros: %.2f%%' % (fail/total*100))

# Loading data
data = get_data('./data/amazon_cells_labelled.txt')
data += get_data('./data/imdb_labelled.txt')
data += get_data('./data/yelp_labelled.txt')


data = adding_data_list(data)
data_training, data_validate = separate_data(data)
classificator = training(data, vectorizer)
analyze_data_validate(data_validate)
