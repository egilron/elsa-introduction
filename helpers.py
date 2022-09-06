from transformers import AutoTokenizer, AutoModelForSequenceClassification,  AdamW
from transformers import BertTokenizer,DistilBertTokenizerFast
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import json
import numpy as np
import random
from sklearn.metrics import accuracy_score


# Helpers for seq classification, from sentence_clf_pt_ds.ipynb

def conll_to_sents(path, separator = "\t"):
    """ Create list of dicts, one per sentence with tokens:list, tags:list and header:str
    This function was redefined here, based on previous versions
    """
    sentences = []
    #Per sentence:

    # returns list of tuples with two lists
    with open (path, 'r') as file:
      textfile = file.read().strip()
    for sent in textfile.split("\n\n"):
      sentencetokens = []
      sentencetags = []
      sentenceheader = ""
      sentlines = sent.strip().split("\n")
      if sentlines[0][0] == "#" and separator not in sentlines[0]:
        sentenceheader = sentlines.pop(0)
      for line in sentlines:
        line = line.strip()
        if separator in line:
          splitline = line.split(separator)
          sentencetokens.append(splitline[0])
          sentencetags.append(splitline[-1])
        else:
          print("Error parsing conll-data in conll_to_sents")
          print(line)
      sentences.append({"tokens": sentencetokens, "tags": sentencetags, "header": sentenceheader})


    return sentences

def has_something(tagslists):
  # For each list (sentence) in the list, check if there are any tokens except "O"
  binary = []
  for t_list in tagslists:
    if set(t_list) == set("O"):
      binary.append(0)
    else:
      binary.append(1)
  return binary




def compute_metrics(pred):
    # Kanskje splitte og ha mer kontroll på denne
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }


def token_data (text):
    '''
    split by space and add start and end indexes 
    '''
    # spaces = [i for i, c in enumerate(text) if c = " "]
    tokens = [{"token": t} for t in text.split(" ")]
    i = 0 # Dataset idenxes seem to start with 1
    for idx, token in enumerate(tokens):
        token["start"] = i
        token["end"] = token["start"] + len(token["token"])
        token["idx"] = idx
        i = token["end"]+1
    return tokens

def conn_tolist(sentence: str, sep=" "):
  '''Takes in a sentence in conll format, like bmes-files norne, where token and tag are separated with <sep>, expect space
  returns one list of tokens, and one list of tags '''
  sentence = sentence.strip()
  assert "\n\n" not in sentence # sentences should be split before this function
  tokens, tags = [], []
  for line in sentence.split("\n"):
    assert line.count(sep) == 1
    token, tag = line.split(sep)
    tokens.append(token)
    tags.append(tag)
  return tokens, tags

def conn_splitsents(text):
  """Splits a text by double line breaks into a list of sentences"""
  return text.strip().split("\n\n")

def tag_one(pred_tags, token_data, single_pred):
  '''receive the pred_tags: list of tags: str
    text: One single string
    single_pred: a dict from the ner pipeline predictions 
    return the pred_tags with sequence from this prediction added'''
  # ent_group = single_pred["entity_group"]
  # start = single_pred["start"]
  # end = single_pred["end"]
  element_tokens = []# Fill with token_idx for those in element 

  for token in token_data: # FInd index of tokens that need to be tagged
    if token["start"] >= single_pred["start"] and token["end"] <= single_pred["end"]: 
        element_tokens.append(token["idx"])
  if len(element_tokens) > 0: # like when token is 'Borten-biografi' and entity returned is "Borten"
    # Tag first token
    pred_tags[min(element_tokens)] = "B-"+ single_pred["entity_group"]
    # Tag subsequent tokens
    for j in range(min(element_tokens)+1, max(element_tokens)+1):
        pred_tags[j] = "I-"+ single_pred["entity_group"]
  
  # New tags are added directly in tags list
  return pred_tags


def pred_ranges(preds: list[dict]):
    # tag is either entity or entity_group
    spans = []
    for pred in preds:
        tag_label = [k for k in pred if k.startswith("entity")][0]
        spans.append({"start": pred["start"], "end": pred["end"], "tag": pred[tag_label],"text": pred["word"] })
    return spans

     
def element_extremes(segments: list):
    """Takes a list of text span values and returns
    a tuple with min, max within that span. Although the spans are sorted, 
    we do not assume they are.
    example: ['0:8', '28:39', '69:72'] returns (0,72)
    EMpty lists return (0,0)"""
    indices = [] # Gets all integers in the list of lists
    for segment in segments:
        indices += [int(n) for n in segment.split(":")]
    if len(indices) < 2:
        return [0,0]
    return [min(indices), max(indices)]

def tag_span(tagslist):
  """Receives a list of BIO tags, 
  returns a list of tuples (label, tagsspan) 
  label is what comes after B or I
  tagsspan is a list of the indices that tag covers
  """
  ongoing = False
  tagsdata = []
  label = ""
  tagsspan = []

  if all([len(t) == 1 for t in tagslist]):
    return [] # pure B and I tags are not to be used in tag_span

  for idx, tag in enumerate(tagslist):
    if tag.startswith("B"):
      if ongoing: #B comes strainght after another tag
        tagsdata.append((label, tagsspan))
      label = tag[2:]
      tagsspan = [idx]
      ongoing = True
    elif tag.startswith("I"):
      tagsspan.append(idx)
    elif tag.startswith("O"):
      if ongoing:
        tagsdata.append((label, tagsspan))
      ongoing = False
  if ongoing:
    tagsdata.append((label, tagsspan))
  
  return tagsdata


def merge_tags(ne_tagspans, tsa_tagspans):
    """receive two outputs from tag_span
    (list of (label,list of indices in span))
    return a unified list according to merge rules"""
    merged_spans = []
    for ne_label, ne_span in ne_tagspans:
      elsa_span = (ne_label+"-Neutral", ne_span) # If no tsa overlap
      for tsa_label, tsa_span in tsa_tagspans:
          tsa_pol = tsa_label.split("-")[-1] # Keep only polarity
          if set.intersection(set(ne_span), set(tsa_span)):
              elsa_span = (ne_label+"-"+tsa_pol, ne_span)
              break 
      merged_spans.append(elsa_span)
        
    return merged_spans

def spans_to_list(startlist, tagspans):
  """Takes an empty ("O") list and a list of tagspans
      tags the empty list accordingly """
  for idx in range(len(startlist)):
    for label, span in tagspans:
      if idx in span:
        prefix = "I-"
        if idx == min(span):
          prefix = "B-"
        startlist[idx] = prefix+label
        break
  return startlist

def compresstags(tagslist):
  """Takes a list where spans are dual, like 'B-ORG-Positive' and removes the centre.
  Returns like B-Positive"""
  compressed = []
  for tag in tagslist:
    splitted = tag.split("-")
    if len(splitted) > 2:
      splitted = [splitted[0], splitted[-1]]
    compressed.append("-".join(splitted))
  return compressed



