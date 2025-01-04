# Assignment 2

Advanced NLP Assignment 2 - Submitted by Ashna Dua, 2021101072

## Theory Questions

- Present in report


## Implementation and training of Transformer

- **`preprocess.py`**: Script for cleaning and tokenizing data

- **`utils.py`**: Script for creation of dataset, dataloaders, Positional Encoding, Feed Forward NN, and other important classes required for training the transformer.

- **`encoder.py`**: Contains Encoder Layer class

- **`decoder.py`**: Contains Decoder Layer class

- **`transformer.py`**: Contains consolidated Transformer class

- **`train.py`**: Script for training model from scratch
**Usage:**
  - **Train from Scratch**: `python train.py`
    - Preprocesses data, trains the model, and saves the model, dataset and vocabularies.

- **`test.py`**: Script for reloading already trained model
**Usage:**
  - **Reload Model**: `python test.py`
    - Reloads the model and displays the BLEU scores on validation and test sets.

- **`calculate_bleu_score.py`**: Script for evaluating model using BLEU metric

Models & Other Data uploaded at: https://drive.google.com/drive/folders/1cL-DRiYr8R2Y1zgEze80lZmNgv-ObEXh?usp=sharing