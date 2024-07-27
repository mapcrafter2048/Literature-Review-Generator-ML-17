## T5-Based Literature Review Generator

### Overview

The Literature Review Generator is a comprehensive tool designed to assist researchers and scholars in generating summaries, extracting keywords, and providing audio summaries for academic and technical documents. It leverages state-of-the-art machine learning and natural language processing techniques to analyze large volumes of text, summarizing content and presenting it in a more accessible format. This tool is particularly useful for literature reviews, enabling users to quickly grasp the main points of extensive documents and articles.

### Features

1. *Summarization*:
   - Automatically generates concise summaries of long documents using a transformer-based model (T5-small).
   - Utilizes a sliding window approach to handle documents exceeding the model's maximum input length.

2. *De-duplication*:
   - Removes duplicate content from the generated summaries using TF-IDF and cosine similarity.

3. *Audio Summaries*:
   - Converts text summaries into audio files, making it easier to consume content on the go.
   - Supports multiple languages for text-to-speech conversion using gTTS.

4. *Keyword Extraction*:
   - Identifies and extracts key phrases from the text using RAKE (Rapid Automatic Keyword Extraction).
   - Generates word clouds to visually represent the most important keywords.

### Installation

To use the Literature Review Generator, you will need to install the required packages. You can do this by running:

bash
pip install transformers datasets torch sklearn pyttsx3 gtts rake-nltk wordcloud matplotlib

## Dataset

The project uses the *['scillm/scientific_papers-archive'](https://huggingface.co/datasets/scillm/scientific_papers-archive)* dataset. Ensure you have access to this dataset, which can be loaded using the datasets library.


### Usage

1. *Data Loading*:
   - Load your dataset using datasets library or prepare your custom dataset in a format compatible with Hugging Face transformers.

2. *Model Training*:
   - Fine-tune the T5-small model on your dataset for custom summarization tasks.

3. *Generating Summaries*:
   - Use the generate_review function to create summaries for given text inputs.

4. *Keyword Extraction and Word Cloud*:
   - Extract keywords using extract_keywords and generate a word cloud using generate_wordcloud.

5. *Audio Summary*:
   - Convert text summaries into audio using text_to_speech.

### Example

python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Example text
text = "Your long academic or technical text goes here."

# Generate summary
summary = generate_review(text, model, tokenizer)
print("Summary:", summary)

# Extract keywords
keywords = extract_keywords(summary)
print("Keywords:", keywords)

# Generate word cloud
generate_wordcloud(keywords)

# Convert summary to audio
audio_file = text_to_speech(summary)
Audio(audio_file)


### Model Training and Evaluation

The model can be fine-tuned on a custom dataset using the provided training loop. The training and evaluation functions handle batching, data loading, and optimization. The example script shows how to prepare the dataset, create a custom DataLoader, and train the model over multiple epochs.

### Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss new features or improvements.

### Acknowledgments

This project uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library for natural language processing tasks and the [datasets](https://github.com/huggingface/datasets) library for dataset management. Special thanks to all the contributors to these open-source projects.
