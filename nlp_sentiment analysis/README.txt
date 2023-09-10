# NAMES
* Amine ZAAMOUN
* Amaury CORNELUS
* Marouan JOUAIDI
* Haiwei FU

# DESCRIPTION OF THE IMPLEMENTED CLASSIFIER
The implemented classifier in this task is a BERT-based model using PyTorch and the transformers library. BERT stands for Bidirectional Encoder Representations from Transformers and is a pre-trained model that can be fine-tuned for specific NLP tasks.

The input to the classifier is a sentence, an aspect term occurring in the sentence, and its aspect category, and it produces a polarity label: positive, negative, or neutral. To encode the input text, the BERT tokenizer is used, which tokenizes the text and converts it into numerical inputs that can be fed into the model.

The `AspectTermDataset` class loads and processes data from the given file. Each input is tokenized using the BERT tokenizer, and the label is encoded as follows: positive is encoded as 0, negative is encoded as 1, and neutral is encoded as 2.

The model is trained using the `train()` method, which loads the training and development data, sets up an AdamW optimizer, and trains the model using the CrossEntropyLoss function. The `predict()` method is used to predict class labels for the input instances in the given file. It loads the test data, runs the model on the data, and returns the list of predicted labels.

The AspectTermDataset class also contains the `pad_collate()` function, which is used to pad the input tensors so that they are of equal length, and to stack them into batches to speed up training.

The tester.py file contains the `train_and_eval()` function, which trains the classifier on the training data, evaluates it on the development set, and evaluates it on the test set (if a test file is provided).

The accuracy obtained on the development dataset is 84.47%, with a standard deviation of 1.32, for the 5 runs in a total execution of 1590.13s (318s per run).

# USAGE
1. Create the Python environment:
python -m venv nlp_env

2. Activate the virtual environment
nlp_env\Scripts\activate

3. Install the required libraries:
pip install torch==1.13.1 pytorch-lightning==1.8.1 transformers==4.22.2 datasets==2.9.0 sentencepiece==0.1.97 scikit-learn==1.2.0 numpy==1.23.5 pandas==1.5.3 nltk==3.8.1 stanza==1.4.2



