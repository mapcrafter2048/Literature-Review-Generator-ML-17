# T5-Based Literature Review Generator

This project utilizes the T5 (Text-To-Text Transfer Transformer) model to generate literature reviews for scientific papers. It includes code for training the model, evaluating its performance, and generating text-based literature reviews.

## Features

- Utilizes the T5 model for generating literature reviews.
- Handles long documents using a sliding window approach.
- De-duplicates generated reviews using TF-IDF and cosine similarity.

## Installation

Ensure you have the following Python packages installed:

```bash
pip install torch transformers datasets scikit-learn tqdm
```

## Dataset

The project uses the `scillm/scientific_papers-archive` dataset. Ensure you have access to this dataset, which can be loaded using the `datasets` library.

## Usage

### Training the Model

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/t5-literature-review-generator.git
    cd t5-literature-review-generator
    ```

2. Run the training script:

    ```bash
    python train.py
    ```

   This script will train the T5 model and save it to the specified directory.

### Generating Literature Reviews

1. Load the trained model and tokenizer:

    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    model = AutoModelForSeq2SeqLM.from_pretrained('path/to/saved/model')
    tokenizer = AutoTokenizer.from_pretrained('path/to/saved/tokenizer')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ```

2. Use the `generate_review` function to generate literature reviews:

    ```python
    def generate_review(text, model, tokenizer, max_input_length=512, max_target_length=128):
        # Implementation as provided in the project
        pass

    text = """Your text here"""
    review = generate_review(text, model, tokenizer)
    print(review)
    ```

### Saving and Loading the Model

To save the model:

```python
import pickle
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('path/to/saved/model')
pickle.dump(model, open('path/to/save/Literature_Review_Generator.pkl', 'wb'))
```

To load the model:

```python
import pickle
from transformers import AutoModelForSeq2SeqLM

model = pickle.load(open('path/to/save/Literature_Review_Generator.pkl', 'rb'))
```

## Contributing

Feel free to fork the repository and submit pull requests. Please open issues if you find bugs or have suggestions for improvements.

## Acknowledgments

- Hugging Face Transformers and Datasets libraries.
- PyTorch.
- The contributors to the `scillm/scientific_papers-archive` dataset.
