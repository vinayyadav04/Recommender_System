from csv import reader
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import xlsxwriter 

path=r"files/domain_classify.csv"
embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=1190
test_portion=.1
corpus = []

sentences=[]
label=[]
labels=[]
with open(path, encoding='windows-1252') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        content0 = row[0]
        content1 = row[1]
        sentences.append(content1)
        label.append(content0)
 
 
for k in label:
  if k=='management':
    labels.append(0)
  elif k=='security':
    labels.append(1)
  elif k=='web development':
    labels.append(2)
  elif k=='coding':
    labels.append(3)
  elif k=='hardware':
    labels.append(4)
  elif k=='higher education':
    labels.append(5)
  elif k=='iot':
    labels.append(6)
  elif k=='java':
    labels.append(7)
  elif k=='artificial intelligance':
    labels.append(8)
  elif k=='python':
    labels.append(9)
  elif k=='finance':
    labels.append(10)
  elif k=='mobile application':
    labels.append(11)
  elif k=='c++':
    labels.append(12)
  elif k=='software':
    labels.append(13)
  elif k=='cloud computing':
    labels.append(14)
  elif k=='networking':
    labels.append(15)
  elif k=='javascript':
    labels.append(16)
  elif k=='machine learning':
    labels.append(17)
  elif k=='blockchain':
    labels.append(18)
  elif k=='datascience':
    labels.append(19)
  else:
    labels.append(20)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size=len(word_index) 
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
split = int(test_portion * training_size)
test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]


embeddings_index = {};
with open(r'files/weights.txt',encoding='utf8') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length,weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(80,activation='relu'),
    tf.keras.layers.Dense(21, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 100

 
training_padded = np.array(training_sequences)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences)
testing_labels = np.array(test_labels)
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2) 
print("Training Complete")



path2=r'files/CCMLEmployeeData.csv'
Name=[]
Domain=[]
event1=[]
event2=[]
with open(path2, encoding='windows-1252') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        content0 = row[0]
        content1 = row[1]
        content2 = row[2]
        content3 = row[3]
        Name.append(content0.lower())
        Domain.append(content1.lower())
        event1.append(content2.lower())
        event2.append(content3.lower())


path3=r'files/event_classify.csv'
training_size1=1000

sentences1=[]
label1=[]
labels1=[]
with open(path3, encoding='windows-1252') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        content0 = row[0]
        content1 = row[1]
        sentences1.append(content1)
        label1.append(content0)
  
for k in label1:      
  if k=='jobs':
    labels1.append(0)
  elif k=='certifications':
    labels1.append(1)
  elif k=='internships':
    labels1.append(2)
  elif k=='competitions':
    labels1.append(3)
  elif k=='workshops':
    labels1.append(4)
  elif k=='trainings':
    labels1.append(5)
  elif k=='seminars':
    labels1.append(6)
  elif k=='hackathons':
    labels1.append(7)
  elif k=='fests':
    labels1.append(8)
  elif k=='webnairs':
    labels1.append(9)
  elif k=='courses':
    labels1.append(10)
  else:
    labels1.append(11)


tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(sentences1)
word_index1 = tokenizer1.word_index
vocab_size1=len(word_index1) 
sequences1 = tokenizer.texts_to_sequences(sentences1)
padded1 = pad_sequences(sequences1, maxlen=max_length, padding=padding_type, truncating=trunc_type) 
split1 = int(test_portion * training_size1) 
test_sequences1 = padded[0:split1]
training_sequences1 = padded[split1:training_size1]
test_labels1 = labels1[0:split1]
training_labels1 = labels1[split1:training_size1]


model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, trainable=False),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(80,activation='relu'),
    tf.keras.layers.Dense(11, activation='softmax')
])
model1.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 100

training_padded1 = np.array(training_sequences1)
training_labels1 = np.array(training_labels1)
testing_padded1 = np.array(test_sequences1)
testing_labels1 = np.array(test_labels1)
 
history = model1.fit(training_padded1, training_labels1, epochs=num_epochs, validation_data=(testing_padded1, testing_labels1), verbose=2)
print("Training Complete")


lis1=[ 'jobs','certifications','internships','competitions','workshops','trainings','seminars','hackathons','fests','webnairs','courses']
lis=['management','security','web development','coding','hardware','higher education','iot','java','artificial intelligance','python','finance','mobile application','c++','software','cloud computing','networking','javascript','machine learning','blockchain','data science','none']


workbook = xlsxwriter.Workbook(r'OUTPUT.xlsx') 
worksheet = workbook.add_worksheet() 
bold = workbook.add_format({'bold': True, 'font_color': 'red'}) 
bolds = workbook.add_format({'bold': True, 'font_color': 'blue'}) 
path4 = r'INPUT.txt'
with open(path4,encoding='utf8') as fp: 
    Lines = fp.readlines() 
    row=1
    rows=1
    worksheet.write(0,0,"EVENTS",bold)
    worksheet.write(0,1,"RECOMMENDED NAMES",bold)
    for line in Lines:
      seed_text = line
      worksheet.write(row,0,seed_text) 
      row=row+1
      token_list = tokenizer.texts_to_sequences([seed_text])[0]
      token_list = pad_sequences([token_list], maxlen=10, padding='pre')
      predicted = model.predict(token_list, verbose=0)
      predicted1 = model1.predict(token_list, verbose=0)      
      domain=lis[np.argmax(predicted)]
      event=lis1[np.argmax(predicted1)]
      a=[]
      for i  in range(len(Domain)):
        if Domain[i]==domain and event1[i]==event or Domain[i]==domain and event2[i]==event:
          a.append(Name[i])
      col=1
      for d in a:
        worksheet.write(rows,col,d,bolds)
        col=col+1    
      rows=rows+1     
workbook.close()

