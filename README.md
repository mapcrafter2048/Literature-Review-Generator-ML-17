# T5-Based Summarization Model for Scientific Papers

This project utilizes the T5 (Text-To-Text Transfer Transformer) model to generate summaries for scientific papers. The model is trained on a dataset of scientific papers, with preprocessing steps to handle long documents and de-duplicate content. This repository includes code for training, evaluation, and generating reviews from text.

## Project Overview

- **Dataset**: Uses the `scillm/scientific_papers-archive` dataset.
- **Model**: `t5-small` model from the Hugging Face Transformers library.
- **Frameworks**: PyTorch, Hugging Face Transformers, Datasets library.
- **Features**: 
  - Sliding window approach for handling long documents.
  - De-duplication of generated reviews using TF-IDF and cosine similarity.
  - Custom dataset and dataloader implementations.
  - Training and evaluation loops.

## Prerequisites

Ensure you have the following Python packages installed:

- `torch`
- `transformers`
- `datasets`
- `sklearn`
- `tqdm`
- `pickle` (standard library)

You can install the required packages using `pip`:

```bash
pip install torch transformers datasets scikit-learn tqdm
```

## Dataset

The project uses the `scillm/scientific_papers-archive` dataset. To load and preprocess the dataset, the code assumes the dataset is available in the `datasets` library.

## Usage

### Training the Model

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/t5-summarization.git
    cd t5-summarization
    ```

2. Run the training script:

    ```bash
    python train.py
    ```

   The training script will:
   - Load and downsample the dataset.
   - Preprocess the data using sliding window and padding.
   - Train the T5 model on the dataset.
   - Save the trained model to a specified directory.

### Generating Summaries

After training the model, you can use it to generate summaries for new texts.

1. Ensure the model and tokenizer are loaded:

    ```python
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    model = AutoModelForSeq2SeqLM.from_pretrained('path/to/saved/model')
    tokenizer = AutoTokenizer.from_pretrained('path/to/saved/tokenizer')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ```

2. Use the `generate_review` function to generate summaries:

    ```python
    def generate_review(text, model, tokenizer, max_input_length=512, max_target_length=128):
        # Implementation as provided in the project
        pass

    text = """Your text here"""
    review = generate_review(text, model, tokenizer)
    print(review)
    ```

### Saving and Loading the Model

The trained model can be saved and loaded using `pickle`:

```python
import pickle
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('path/to/saved/model')
pickle.dump(model, open('path/to/save/Review_Generator.pkl', 'wb'))
```

To load the model:

```python
import pickle
from transformers import AutoModelForSeq2SeqLM

model = pickle.load(open('path/to/save/Review_Generator.pkl', 'rb'))
```

## Notes

- The `train.py` script is designed to handle large datasets by downsampling and preprocessing. Adjust `DATASET_SIZE`, `SAMPLE_FRACTION`, and other constants based on your requirements.
- Ensure that GPU support is enabled if you are working with large datasets and models to speed up training and evaluation.

## Contributing

Feel free to fork the repository, submit pull requests, or open issues if you have suggestions or encounter problems.

## Acknowledgments

- Hugging Face Transformers and Datasets libraries.
- PyTorch for deep learning.
- The contributors to the `scillm/scientific_papers-archive` dataset.

---

