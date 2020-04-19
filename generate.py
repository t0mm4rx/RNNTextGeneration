import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

file = open("victorhugo.txt").read()

file = file.replace('\n \n', '\n')
file = file.replace('  ', ' ')
def tokenize_words(input):
    # lowercase everything to standardize it
    # input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    # filtered = filter(lambda token: token not in stopwords.words('french'), tokens)
    return " ".join(tokens)

# preprocess the input data, make tokens
processed_inputs = tokenize_words(file)

print(processed_inputs)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)

print(chars)


seq_length = 100
x_data = []
y_data = []

# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

model = load_model('model')

num_to_char = dict((i, c) for i, c in enumerate(chars))

start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

res = ""
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    # print(prediction, index)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    res += result

    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("Res: " + res)
