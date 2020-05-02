# Neural Machine translation with a Seq2seq model 

Welcome to our Seq2seq repository! 
The goal of the project is to implement a NMT model to perform translation from French to English. To do so, we relied on a PyTorch tutorial written by Sean Robertson : [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).  

The project is an academic project from [CentraleSupélec](https://www.centralesupelec.fr/) 
during the NLP course.  

## Getting started

```
pip install -r requirement.txt 
jupyter notebook seq2seq_notebook.ipynb 
```

We relied on the model provided in the PyTorch tutorial. We trained it on [datasets](https://wit3.fbk.eu/mt.php?release=2014-01) recording Ted talks and their translations. We implemented beam search with length penalty used for inference time in a script called `beam_search.py`. 

## Project structure
The structure of our project is as follow : 
 ```
• seq2seq_notebook.ipynb 
• beam_search.py
+ ted_data
  + train 
    • IWSLT14.TED.dev2010.en-fr.en.xml 
    • IWSLT14.TED.dev2010.en-fr.fr.xml
    • IWSLT14.TED.tst2010.en-fr.en.xml
    • IWSLT14.TED.tst2010.en-fr.fr.xml
   + test
    • IWSLT14.TED.tst2011.en-fr.fr.xml
    • IWSLT14.TED.tst2011.en-fr.en.xml 
    • IWSLT14.TED.tst2012.en-fr.en.xml
    • IWSLT14.TED.tst2012.en-fr.fr.xml
```

## Results using Bleu 
Mettre un petit tableau
