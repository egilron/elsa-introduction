# Elsa-introduction
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

## Run the 1_prepare_dataset.ipynb
This will read the TSA conll data and create a sentence-based table in `data_cache` path. Also, each of the text sources are saved to ta text file togeter with information on their named entities and overlapping sentiment targets.

## Run the 2_resolve-entity_level.ipynb
This aggregates the collected data to the entity level. For each volitional entity, the relevant document-level, sentiment-level and target-level sentiment information is registered. This information is shared with the annotator who assigns the entity-level sentiment that the text i total conveys towards the entity. If the annotator considers the reckognized named entity to be wrong, the entity is labelled "spurious".

## Run the 3_analyze_annotated.ipynb
After the manual polarity is assigned for each entity, this notebook reads the results, and counts the hits and misses where the entity is assigned the correct or wrong label be the other sentiment annotations.
