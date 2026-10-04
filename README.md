# Movie Review Sentiment Analysis with DistilBERT

An end-to-end NLP sentiment analysis project that uses DistilBERT to classify movie reviews as **positive or negative**.

The project evaluates a pre-trained transformer model against the full **25,000-review IMDB test dataset**, experiments with fine-tuning for modern slang and informal language, and provides both command-line and Streamlit interfaces for interactive sentiment analysis.

The project also explores practical NLP limitations including ambiguous language, slang, sarcasm, long-text truncation, model confidence, and label-mapping issues encountered during fine-tuning.

---

## Project Objectives

The main objectives of this project were to:

- Implement transformer-based sentiment analysis using Hugging Face Transformers
- Evaluate a pre-trained DistilBERT model against a labelled real-world dataset
- Measure classification performance using accuracy, precision, recall and F1-score
- Fine-tune the model using additional examples containing modern slang and informal language
- Test the model against unseen custom reviews
- Investigate incorrect and unexpected predictions
- Build an interactive command-line sentiment analyser
- Develop a Streamlit application for easier user interaction
- Explore limitations associated with long reviews, ambiguity and sarcasm

---

## Features

- **Transformer-based NLP:** Uses DistilBERT for binary sentiment classification
- **25,000-review evaluation:** Evaluated against the complete IMDB test split
- **Fine-tuning experiment:** Extended the model with additional contemporary and slang-based examples
- **Interactive testing:** Users can enter their own movie reviews
- **Confidence scoring:** Displays model confidence alongside each prediction
- **Streamlit application:** Provides a simple browser-based interface
- **Model evaluation:** Uses accuracy, precision, recall and F1-score
- **Error analysis:** Tests ambiguous, informal and difficult sentiment expressions
- **Long-review handling:** Uses truncation to remain within DistilBERT's token limit

---

## Dataset

The primary evaluation dataset is the IMDB movie review dataset accessed through Hugging Face Datasets.

The test split contains:

| Class | Reviews |
| --- | ---: |
| Negative | 12,500 |
| Positive | 12,500 |
| **Total** | **25,000** |

The balanced class distribution makes accuracy considerably more informative than it would be on a heavily imbalanced dataset.

The dataset was shuffled before evaluation so that predictions were not processed solely in their original class ordering.

---

## Base Model

The project uses:

`distilbert-base-uncased-finetuned-sst-2-english`

This is a DistilBERT model already fine-tuned for binary sentiment classification.

| Property | Value |
| --- | --- |
| Architecture | DistilBERT |
| Task | Binary Sentiment Classification |
| Labels | POSITIVE / NEGATIVE |
| Maximum Input Length | 512 tokens |
| Framework | Hugging Face Transformers |
| Evaluation Dataset | IMDB |
| Test Reviews | 25,000 |

---

## Model Evaluation

The model was evaluated against all **25,000 reviews** in the IMDB test dataset.

### Results

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Negative | 0.87 | 0.92 | 0.89 | 12,500 |
| Positive | 0.91 | 0.86 | 0.89 | 12,500 |
| **Accuracy** | | | **0.89** | **25,000** |
| Macro Average | 0.89 | 0.89 | 0.89 | 25,000 |
| Weighted Average | 0.89 | 0.89 | 0.89 | 25,000 |

### Findings

The DistilBERT model achieved approximately **89% accuracy across 25,000 unseen IMDB movie reviews**, demonstrating strong overall performance for binary sentiment classification.

Performance was relatively balanced between the two sentiment classes, with an F1-score of approximately **0.89 for both positive and negative reviews**.

The model performed particularly well at identifying negative reviews, achieving **92% recall**. This indicates that most genuinely negative reviews were correctly detected.

Positive reviews achieved a higher **91% precision**, meaning that when the model classified a review as positive, the prediction was usually correct. However, positive recall was lower at **86%**, indicating that some genuinely positive reviews were incorrectly classified as negative.

This suggests a modest tendency for the model to classify difficult or ambiguous positive reviews as negative.

Overall, the evaluation demonstrates that the pre-trained DistilBERT model generalises effectively from its original sentiment training to the IMDB movie-review domain, while still producing errors on more linguistically complex examples.

---

## Fine-Tuning Experiment

The project was extended by fine-tuning the model with additional sentiment examples containing informal and contemporary language.

The purpose was not simply to improve headline accuracy, but to investigate whether additional training examples could improve performance on language that may be difficult for a conventional sentiment classifier.

Examples included expressions such as:

- `This movie hits different in a good way`
- `brill`
- `crap`
- other short, informal or slang-based sentiment expressions

Following fine-tuning, interactive testing showed that the model could correctly interpret examples such as:

```text
Enter review: This movie hits different in a good way
→ Sentiment: POSITIVE
→ Confidence: 99.97%
```

```text
Enter review: crap
→ Sentiment: NEGATIVE
→ Confidence: 99.96%
```

```text
Enter review: brill
→ Sentiment: POSITIVE
→ Confidence: 82.71%
```

These tests demonstrate that the fine-tuned model can recognise some informal expressions and slang that may not appear in conventional movie-review language.

### Fine-Tuning Debugging

Fine-tuning also introduced an important practical lesson.

During development, apparently positive reviews were initially being displayed as negative even when the model's raw output was:

```python
{'label': 'POSITIVE', 'score': 0.999...}
```

The issue was therefore not necessarily an incorrect model prediction. The application logic was manually expecting labels such as `LABEL_0` and `LABEL_1`, while the loaded model returned readable `POSITIVE` and `NEGATIVE` labels.

Inspecting the **raw inference output** helped isolate the problem and correct the label-handling logic.

This highlighted an important machine-learning engineering principle: unexpected application output does not automatically indicate poor model performance. The complete inference pipeline — model configuration, label mapping and application code — must also be validated.

---

## Interactive Sentiment Analysis

The project includes a command-line interface that allows users to test their own reviews.

Example:

```text
==================================================
Custom Sentiment Analysis
==================================================
Enter movie reviews to analyze (or 'quit' to exit)

Enter review: This movie hits different in a good way
→ Sentiment: POSITIVE
→ Confidence: 99.97%

Enter review: crap
→ Sentiment: NEGATIVE
→ Confidence: 99.96%

Enter review: brill
→ Sentiment: POSITIVE
→ Confidence: 82.71%
```

This provides a simple way to perform qualitative testing alongside the quantitative IMDB evaluation.

---

## Streamlit Application

A Streamlit interface was developed to make the sentiment model accessible through a browser rather than requiring users to interact directly with the Python command line.

The application allows a user to:

1. Enter a movie review
2. Submit the text to the trained sentiment classifier
3. Receive a POSITIVE or NEGATIVE prediction
4. View the model's confidence score

For example, a long positive review tested through the application produced:

```text
Sentiment: POSITIVE
Confidence: 99.73%
```

The Streamlit implementation demonstrates how a trained NLP model can be incorporated into a simple end-user application rather than existing only as an experimental Python script.

---

## Handling Long Reviews

DistilBERT has a maximum sequence length of **512 tokens**.

During testing, submitting a substantially longer review without appropriate handling resulted in a tensor dimension error because the input exceeded the model's supported sequence length.

The inference pipeline therefore uses:

```python
truncation=True,
max_length=512
```

This prevents oversized inputs from causing the application to fail.

However, truncation introduces an important limitation: text occurring after the first 512 tokens may not contribute to the prediction.

For a long review in which the author's conclusion differs from the beginning of the review, this could result in important sentiment information being discarded.

### Potential Alternative: Sliding Window

A future implementation could:

- Split long reviews into overlapping chunks
- Classify each chunk independently
- Aggregate the probabilities or predictions
- Produce an overall review-level sentiment

This would preserve considerably more information than simple truncation.

---

## Model Limitations

Testing demonstrated several important limitations of binary transformer-based sentiment analysis.

### Ambiguous Language

Short statements such as:

```text
Not good but not bad.
```

can be difficult to classify because the text contains conflicting sentiment signals.

A binary model must still select either POSITIVE or NEGATIVE even when the underlying statement is effectively neutral or mixed.

### Neutral Language

Expressions such as:

```text
Neutral
```

or:

```text
Meh
```

must also be forced into one of the two available classes.

This illustrates an architectural limitation rather than necessarily a model failure: the classifier has no `NEUTRAL` class available.

### Sarcasm

Sarcasm remains particularly challenging.

For example, a statement may contain strongly positive vocabulary while communicating a negative opinion through context:

```text
Great, another two hours of my life I'll never get back.
```

A sentiment model may focus heavily on words such as `great` unless it has learned the contextual pattern expressing sarcasm.

### Confidence Does Not Equal Correctness

Several tests produced extremely high confidence scores.

However, a high confidence score represents the model's confidence in its own classification; it does **not** guarantee that the classification is objectively correct.

This is particularly important for:

- sarcasm
- ambiguous reviews
- slang
- mixed sentiment
- extremely short text
- language outside the model's training distribution

---

## Key Findings

The project produced several important findings:

1. **DistilBERT provides strong baseline performance.**  
   Approximately 89% accuracy was achieved across the full 25,000-review IMDB test set.

2. **Performance was relatively balanced.**  
   Both sentiment classes achieved an F1-score of approximately 0.89.

3. **Negative reviews were detected more consistently.**  
   Negative recall reached 0.92 compared with 0.86 for positive reviews.

4. **Fine-tuning can improve handling of domain-specific language.**  
   Additional slang examples allowed the model to recognise informal expressions such as `brill`, `crap`, and `hits different in a good way`.

5. **Model debugging requires inspection of raw predictions.**  
   An apparent classification problem was traced to application-side label mapping rather than simply assuming the fine-tuned model was incorrect.

6. **Binary sentiment classification has inherent limitations.**  
   Neutral, mixed and ambiguous statements must still be assigned to one of two classes.

7. **Sarcasm remains difficult.**  
   Literal vocabulary can conflict with the intended sentiment, making contextual interpretation challenging.

8. **Confidence scores require careful interpretation.**  
   High confidence does not guarantee semantic correctness.

9. **Long text requires explicit handling.**  
   DistilBERT's 512-token input limit means truncation or chunking is required for lengthy movie reviews.

10. **Deployment changes the nature of the project.**  
    Building the Streamlit interface demonstrates how an NLP model can move from experimentation and evaluation into a usable application.


## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd sentiment-analysis
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Command-Line Application

```bash
python sentiment_analyser-finetuned.py
```

### Streamlit Application

```bash
streamlit run sentiment_app.py
```

---

## Requirements

```text
streamlit
transformers
torch
datasets
scikit-learn
pandas
numpy
```

---

## Future Improvements

Potential extensions include:

- Implement a sliding-window approach for reviews exceeding 512 tokens
- Introduce a third `NEUTRAL` sentiment class
- Build a larger and more systematically balanced fine-tuning dataset
- Create dedicated evaluation sets for slang, sarcasm and ambiguous language
- Compare base-model and fine-tuned-model performance quantitatively
- Generate and visualise a confusion matrix
- Add batch prediction from CSV files
- Add aspect-based sentiment analysis for areas such as acting, plot and cinematography
- Investigate multilingual sentiment classification
- Expose the model through an API
- Compare DistilBERT against alternative transformer architectures

---

## Skills Demonstrated

This project demonstrates practical experience with:

- Natural Language Processing (NLP)
- Transformer models
- DistilBERT
- Hugging Face Transformers
- Hugging Face Datasets
- Transfer learning and fine-tuning
- PyTorch
- Model inference
- Binary classification
- Precision, recall and F1-score
- Model evaluation
- Error analysis
- Tokenisation and sequence-length constraints
- Model debugging
- Python
- Streamlit
- Interactive AI application development

---

## License

MIT License.

---

## Acknowledgements

- Hugging Face for the Transformers and Datasets libraries
- Stanford / IMDB dataset contributors for the movie-review dataset
- Streamlit for the web application framework
- IT Online Learning for the guided project and training material
