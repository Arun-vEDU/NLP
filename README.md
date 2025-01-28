# Model Comparison and Analysis

Below is a comparison table for different models:

| Model              | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy |
|---------------------|-------------|---------------|---------------|---------------------|-------------------|
| Skipgram           | 2          | 22.80             | 11 msec/Epoch             | -                   | -                 |
| Skipgram (NEG)     | 2           | 27.50            | 4 msec/Epoch            | -                   | -                 |
| Glove              | 2           | -             | -             | -                   | -                 |
| Glove (Gensim)     | 2          | 0            | 5.85 sec/Epoch            | 0                  | -                 |
