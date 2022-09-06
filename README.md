# elsa-introduction
ELSA data exploration and modelling as presented in the COLING 2022 paper "Entity-Level Sentiment Analysis (ELSA): An exploratory task survey"

# Obtain the data
Our experiments are run on Norwegian newspaper reviews, annotated for various levels of sentiment.
The data are openly available on github. 
We need three levels of sentiment annotations: 
- Document-level sentiment classification 
- Sentence-level sentiment classification
- Target-level sentiment classifications

The data can be obtain from the following repositories, and we suggest you place them as subfolder to this root folder.
```
git clone https://github.com/ltgoslo/norec.git
git clone https://github.com/ltgoslo/norec_tsa.git
git clone https://github.com/ltgoslo/norec_sentence.git
```

## Run the prepare_dataset.ipynb
This will read the TSA conll data and create a sentence-based table in `data_cache`path.

## Run the doc_sentence_analyze.ipynb
This extracts documentwise and sentencewise sentiment information