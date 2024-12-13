
#%% FJIRST BERT MODEL
import pandas as pd
file_path ="D:\Medical-Abstracts-TC-Corpus-main\medical_tc_train.csv"
train_data = pd.read_csv(file_path)

test_file_path ="D:\Medical-Abstracts-TC-Corpus-main\medical_tc_test.csv"
test_data = pd.read_csv(test_file_path)
#%%
import numpy as np
import pandas as pd
file_path =("D:\Medical-Abstracts-TC-Corpus-main\medical_tc_labels.csv")
labels = pd.read_csv(file_path)['condition_name'].values
labels
#%%
train_data['medical_abstract'][80]
#%%
train_data['condition_label'].values
#%%
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("text-classification", model="sid321axn/Bio_ClinicalBERT-finetuned-medicalcondition")

#%%
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForTokenClassification
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenizer = AutoTokenizer.from_pretrained("sid321axn/Bio_ClinicalBERT-finetuned-medicalcondition")
# model = AutoModelForSequenceClassification.from_pretrained("sid321axn/Bio_ClinicalBERT-finetuned-medicalcondition")
#%%
test_data['medical_abstract'].values
#%% preprocessed_train_data
train_sentenses=[s for s in train_data['medical_abstract'].values]
inputs = tokenizer(train_sentenses, padding=True, truncation=True ,max_length=512, return_tensors="pt") 
preprocessed_train_data=inputs
preprocessed_train_data['input_ids'][5].shape
#%% preprocessed_test_data
test_sentenses=[s for s in test_data['medical_abstract'].values]
test_inputs = tokenizer(test_sentenses, padding=True, truncation=True ,max_length=512, return_tensors="pt") 
preprocessed_test_data=test_inputs
#%%

# results=model(**preprocessed_test_data)
#%%
from datasets import Dataset, DatasetDict

# Define your data
data = {
    'labels': train_data['condition_label'].values ,
    'input_ids': preprocessed_train_data['input_ids'],
    'token_type_ids': preprocessed_train_data['token_type_ids'],
    'attention_mask':preprocessed_train_data['attention_mask']
}

test_data= {
    'labels': test_data['condition_label'].values ,
    'input_ids': preprocessed_test_data['input_ids'],
    'token_type_ids': preprocessed_test_data['token_type_ids'],
    'attention_mask':preprocessed_test_data['attention_mask']
}
#%%
# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': Dataset.from_dict(data)
    , 'test':Dataset.from_dict(test_data),
}) 

# Print dataset information
print(dataset_dict)


# %%
from transformers import AutoTokenizer

from transformers import pipeline
# %%
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="./test_trainer")
# %%
import numpy as np
import evaluate

metric = evaluate.load("accuracy","loss")
# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="./test_trainer", evaluation_strategy="epoch", num_train_epochs=10)
# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    compute_metrics=compute_metrics,
)
# %%
results=trainer.train()
# %%

import torch

predictions = torch.nn.functional.softmax(results.logits, dim=-1)
print(predictions)
#%%
model.config.id2label
#%%
predictions.detach().numpy()
#%%
predictions.shape


# %%  
# del model
# del trainer
import torch
torch.cuda.empty_cache()

# tokenizer = AutoTokenizer.from_pretrained("sid321axn/Bio_ClinicalBERT-finetuned-medicalcondition")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)#, ignore_mismatched_sizes=True)
model = AutoModelForSequenceClassification.from_pretrained("sid321axn/Bio_ClinicalBERT-finetuned-medicalcondition")#, num_labels=5, ignore_mismatched_sizes=True)

# model.config.num_hidden_layers = 1
#%%
dataset_dict.set_format("torch")
# %%
small_train_dataset = dataset_dict["train"].shuffle(seed=42)#.select(range(1000))
small_eval_dataset = dataset_dict["test"].shuffle(seed=42)#.select(range(1000))
# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)
# %%
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
# %%
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
# %%
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %%
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
# %%
import evaluate

metric = evaluate.load("BucketHeadP65/confusion_matrix")
model.eval()
predicts=[]
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    for p in predictions:
        predicts.append(p)
cf=metric.compute()
# %%
torch.save(model, './BioMedicalBert.pt')
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

conf_matrix_percentage = cf['confusion_matrix'] / cf['confusion_matrix'].sum(axis=1, keepdims=True)

plt.figure(figsize=(5, 3))
ConfusionMatrixDisplay(conf_matrix_percentage).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Count the number of parameters
num_parameters = count_parameters(model)
print("Number of parameters:", num_parameters) #Number of parameters: 108317193






# %% Second BERT MODEL

from transformers import pipeline

pipe = pipeline("text-classification", model="Humberto/MedicalArticlesClassificationModelMultiLabel")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Humberto/MedicalArticlesClassificationModelMultiLabel")
model = AutoModelForSequenceClassification.from_pretrained("Humberto/MedicalArticlesClassificationModelMultiLabel",from_tf=True)
# %%
import pandas as pd
from datasets import Dataset, DatasetDict
file_path ="D:\Medical-Abstracts-TC-Corpus-main\medical_tc_train.csv"
train_data = pd.read_csv(file_path)

test_file_path ="D:\Medical-Abstracts-TC-Corpus-main\medical_tc_test.csv"
test_data = pd.read_csv(test_file_path)

train_sentenses=[s for s in train_data['medical_abstract'].values]
inputs = tokenizer(train_sentenses, padding=True, truncation=True ,max_length=512, return_tensors="pt") 
preprocessed_train_data=inputs
preprocessed_train_data['input_ids'][5].shape
#%% preprocessed_test_data
test_sentenses=[s for s in test_data['medical_abstract'].values]
test_inputs = tokenizer(test_sentenses, padding=True, truncation=True ,max_length=512, return_tensors="pt") 
preprocessed_test_data=test_inputs
#%%
import torch
# Define your data
data = {
    'labels': train_data['condition_label'].values,
    'input_ids': preprocessed_train_data['input_ids'],
    # 'token_type_ids': preprocessed_train_data['token_type_ids'],
    'attention_mask':preprocessed_train_data['attention_mask']
}

test_data= {
    'labels': test_data['condition_label'].values ,
    'input_ids': preprocessed_test_data['input_ids'],
    # 'token_type_ids': preprocessed_test_data['token_type_ids'],
    'attention_mask':preprocessed_test_data['attention_mask']
}
#%%
from transformers import TrainingArguments, Trainer

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': Dataset.from_dict(data)
    , 'test':Dataset.from_dict(test_data),
}) 

training_args = TrainingArguments(output_dir="./test_trainer2")
dataset_dict.set_format("torch")
from tqdm.auto import tqdm


small_train_dataset = dataset_dict["train"].shuffle(seed=42)#.select(range(1000))
small_eval_dataset = dataset_dict["test"].shuffle(seed=42)#.select(range(1000))
# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)
# %%
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
# %%
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
# %%
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# %%
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
# %%
import evaluate

metric = evaluate.load("accuracy")#,"loss","precision", "recall","BucketHeadP65/confusion_matrix")
model.eval()
predicts=[]
for batch in eval_dataloader:
    # batch = {k: v.to(device) for k, v in batch.items()}
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    for p in predictions:
        predicts.append(p)
metrics=metric.compute()
#%%
print(metrics)
# %%
torch.save(model, './MedicalArticleBert.pt')
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

conf_matrix_percentage = cf['confusion_matrix'] / cf['confusion_matrix'].sum(axis=1, keepdims=True)

plt.figure(figsize=(5, 3))
ConfusionMatrixDisplay(conf_matrix_percentage).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Count the number of parameters
num_parameters = count_parameters(model)
print("Number of parameters:", num_parameters) # Number of parameters: 66958086

# %%









#%% CNN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



nltk.data.path.append(r"D:/Medical-Abstracts-TC-Corpus-main")  # Replace with your actual NLTK data path


# nltk.data.path.append(r"/content/drive/MyDrive/Medical-Abstracts-TC-Corpus-main")  # Replace with your actual NLTK data path
# Load your data from CSV file
file_path ="D:/Medical-Abstracts-TC-Corpus-main\medical_tc_train.csv"
train_data = pd.read_csv(file_path)

test_file_path ="D:\Medical-Abstracts-TC-Corpus-main\medical_tc_test.csv"
test_data = pd.read_csv(test_file_path)

nltk.download('punkt')
nltk.download('stopwords')
# tokenizer
# Define your preprocessing function
def preprocess_and_get_length(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    # tokens = re.split(r' |-|\(|\)', text)  # Split by space, hyphen, parentheses
    tokens = re.findall(r'\b(?:\w+(?:[.-]\w+)*|[^\w\s])', text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    processed_tokens = []
    for token in tokens:
        if re.match(r'^\d+(\.\d+)?$', token):
            processed_tokens.append(float(token))  # Convert to float if it's a float
        elif token not in stop_words and not re.match(r'^\W+$', token):
            processed_tokens.append(token.lower())

    # Return the length of the processed text
    return processed_tokens

# Apply the preprocessing and get the length for each text
tokenized_data = train_data['medical_abstract'].apply(preprocess_and_get_length)
tokenized_data_test = test_data['medical_abstract'].apply(preprocess_and_get_length)

# Convert tokenized data to strings
tokenized_texts = tokenized_data.apply(lambda x: ' '.join(map(str, x)))
tokenized_texts_test = tokenized_data_test.apply(lambda x: ' '.join(map(str, x)))

# Create a tokenizer and fit on tokenized_data without numbers
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(tokenized_texts)

# Convert tokenized data to sequences without further processing
sequences_train = []
for text in tokenized_texts:
    seq = []
    for token in text.split():
        if token in tokenizer.word_index:
            seq.append(tokenizer.word_index[token])
        else:
            try:
                seq.append(float(token))  # Try converting to int
            except ValueError:
                seq.append(tokenizer.word_index['<OOV>'])  # If not a number, use OOV token
    sequences_train.append(seq)

sequences_test = []
for text in tokenized_texts_test:
    seq = []
    for token in text.split():
        if token in tokenizer.word_index:
            seq.append(tokenizer.word_index[token])
        else:
            try:
                seq.append(float(token))  # Try converting to int
            except ValueError:
                seq.append(tokenizer.word_index['<OOV>'])  # If not a number, use OOV token
    sequences_test.append(seq)

# Pad sequences to ensure consistent length for input to the CNN
max_length = max(max(len(seq) for seq in sequences_train), max(len(seq) for seq in sequences_test))
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define a custom padding function
import tensorflow as tf

def custom_pad_sequences(sequences, maxlen=None, padding='pre', truncating='pre', value=0.):
    # Pad sequences with float value
    # padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating, value=value)

    # Convert the padded sequences to TensorFlow tensors
    seqs_tensors = [tf.convert_to_tensor(seq, dtype=tf.float32) for seq in sequences]
    seqs_tensors2=[]
    # Pad the list of tensors with zeros if needed
    for par in seqs_tensors:
      num_padding = 450 - len(par)
      # print(par)
      if num_padding > 0:
        seqs_tensors2.append( tf.concat([par , [tf.constant(0.0, dtype=tf.float32)] * num_padding], axis=0))
      else:
        print(len(par))

    # Concatenate the tensors along the first dimension
    # padded_seqs_tensor = tf.convert_to_tensor(seqs_tensors)

    return seqs_tensors2

# Usage
padded_train = custom_pad_sequences(sequences_train, maxlen=max_length, padding='post', truncating='post', value=0.)
padded_test = custom_pad_sequences(sequences_test, maxlen=max_length, padding='post', truncating='post', value=0.)
padded_test=np.array(padded_test)
padded_train=np.array(padded_train)
#%%
num_classes=5
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Assuming 'conditional_label' is the column containing your categorical labels
labels = train_data['condition_label']

# Step 1: Convert labels to numerical indices using LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Step 2: Convert numerical indices to one-hot encoded vectors
labels_train_one_hot = tf.keras.utils.to_categorical(labels_encoded)

labels_test = test_data['condition_label']

# Step 1: Convert labels to numerical indices using LabelEncoder
label_encoder_test = LabelEncoder()
labels_encoded_test = label_encoder.fit_transform(labels_test)

# Step 2: Convert numerical indices to one-hot encoded vectors
labels_test_one_hot = tf.keras.utils.to_categorical(labels_encoded_test)

#%%
# Assuming train_data is your DataFrame
labels = train_data['condition_label'].tolist()

# Subtract 1 from each element in the list
labels_minus_one = [label - 1 for label in labels]
# Assuming train_data is your DataFrame
labels_test = test_data['condition_label'].tolist()

# Subtract 1 from each element in the list
labels_minus_one_test = [label - 1 for label in labels_test]

#%% train cnn 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Calculate class weights

class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(labels_minus_one),
                                        y = labels_minus_one
                                    )
class_weights_dict = dict(enumerate(class_weights))


print(f'class_weights_dict : {class_weights_dict}')


# Define the CNN model
embedding_dim = 50
filters = 128

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)*5, output_dim=embedding_dim, input_length=450))

# model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_length))
model.add(Conv1D(filters=filters, kernel_size=5, activation='relu'))  # Kernel Size
model.add(GlobalMaxPooling1D())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model with class weights
custom_optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) #5

# Train the model with early stopping and class weights
model.fit(
    padded_train,
    labels_train_one_hot,
    epochs=100,
    batch_size=64,
    validation_data=(padded_test, labels_test_one_hot),
    callbacks=[early_stopping],
    class_weight=class_weights_dict
)

# Evaluate the model on test data
predictions = model.predict(padded_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(labels_test_one_hot, axis=1)

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_report = classification_report(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)


#%%
# Assuming 'model' is your Keras CNN model
model.save("./MedicalClassificationCNN.keras")

# %%
