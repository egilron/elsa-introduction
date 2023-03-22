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
# Modelling ELSA
Since there presently does not exist a manually annotated dataset for ELSA, we create a proxy dataset by combining the manually annotated [Targeted Sentiment Analysis (TSA)](https://github.com/ltgoslo/norec_tsa) dataset with NER inference. This is done in **4_modelling_dataprep.ipynb**.
This dataset can be used for training a token classification model. Read more in **4_modelling_dataprep.ipynb**

# Cite the paper

```
@inproceedings{ronningstad-etal-2022-entity,
    title = "Entity-Level Sentiment Analysis ({ELSA}): An Exploratory Task Survey",
    author = "R{\o}nningstad, Egil  and
      Velldal, Erik  and
      {\O}vrelid, Lilja",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.589",
    pages = "6773--6783",
    abstract = "This paper explores the task of identifying the overall sentiment expressed towards volitional entities (persons and organizations) in a document - what we refer to as Entity-Level Sentiment Analysis (ELSA). While identifying sentiment conveyed towards an entity is well researched for shorter texts like tweets, we find little to no research on this specific task for longer texts with multiple mentions and opinions towards the same entity. This lack of research would be understandable if ELSA can be derived from existing tasks and models. To assess this, we annotate a set of professional reviews for their overall sentiment towards each volitional entity in the text. We sample from data already annotated for document-level, sentence-level, and target-level sentiment in a multi-domain review corpus, and our results indicate that there is no single proxy task that provides this overall sentiment we seek for the entities at a satisfactory level of performance. We present a suite of experiments aiming to assess the contribution towards ELSA provided by document-, sentence-, and target-level sentiment analysis, and provide a discussion of their shortcomings. We show that sentiment in our dataset is expressed not only with an entity mention as target, but also towards targets with a sentiment-relevant relation to a volitional entity. In our data, these relations extend beyond anaphoric coreference resolution, and our findings call for further research of the topic. Finally, we also present a survey of previous relevant work.",
}
````

