# Kirundi News Topic Classification

NLP Summative Project, African Leadership University.

Classifies Kirundi news articles into six topics: business, entertainment,
health, politics, religion, sports.

## Links
- Live app: https://huggingface.co/spaces/sonia250/kirundi-news-classifier
- Model: https://huggingface.co/sonia250/afriberta-kirundi-news
- Demo video: (coming soon)

## Dataset
MasakhaNEWS, Kirundi subset (`run`), from Hugging Face.
1,117 train / 159 validation / 322 test articles.

## Results (test set)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.863 | 0.776 | 0.859 |
| Word2Vec + BiLSTM | 0.839 | 0.744 | 0.835 |
| AfriBERTa (fine-tuned) | 0.916 | 0.853 | 0.915 |
| AfroXLMR (fine-tuned) | 0.901 | 0.829 | 0.899 |

## Repository structure
- `notebooks/`: data exploration, baselines, AfriBERTa, AfroXLMR, error analysis
- `results/`: figures, metrics, predictions, error list
- `app/`: web app code

## Status
In progress: full methodology and run instructions coming.