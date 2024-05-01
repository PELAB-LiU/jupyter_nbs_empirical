#!/usr/bin/python
import multiprocessing
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
import numpy as np
from gensim.test.utils import datapath

def retrain_word2vec(tokenized_traindata, outputpath, modelname="nberr_word2vec.model", vector_size=100):
    cores = multiprocessing.cpu_count()

    try:
        w2v_model = Word2Vec(vector_size=vector_size, epochs=50, sg=0, min_count=1, workers=cores-1) # CBOW, sg=1 skip-gram
        w2v_model.build_vocab(tokenized_traindata)
        w2v_model.train(tokenized_traindata, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
        
        w2v_model.init_sims(replace=True)
        w2v_model.save(str(outputpath.joinpath(modelname)))
        print('Training has finished. Model saved in file.')
        
        return w2v_model
        
    except Exception as e:
        print('Training model error:', e)

def finetune_word2vec(tokenized_traindata, glove_vectors, outputpath, modelname="nberr_word2vec_glove_finetune.model", vector_size=200):
    cores = multiprocessing.cpu_count()

    try:
        # build a toy model to update with
        w2v_model = Word2Vec(vector_size=vector_size, epochs=50, min_count=1, workers=cores-1)
        w2v_model.build_vocab(tokenized_traindata)

        # add GloVe's vocabulary & weights
        w2v_model.build_vocab([list(glove_vectors.keys())], update=True)

        # train on our data
        w2v_model.train(tokenized_traindata, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)
        
        w2v_model.init_sims(replace=True)
        w2v_model.save(str(outputpath.joinpath(modelname)))
        print('Training has finished. Model saved in file.')
        
        return w2v_model
        
    except Exception as e:
        print('Training model error:', e)
        
def load_word2vec(modelpath, modelname="nberr_word2vec.model"):
    return Word2Vec.load(str(modelpath.joinpath(modelname)))

def retrain_subword2vec(tokenized_traindata, outputpath, vector_size=200, modelname="nberr_subword2vec.model"): # fasttext
    cores = multiprocessing.cpu_count()

    try:
        fb_model = FastText(tokenized_traindata, vector_size=vector_size, epochs=50, sg=0, min_count=1, workers=cores-1)
        
        fb_model.save(str(outputpath.joinpath(modelname)))
        print('Training has finished. Model saved in file.')
        
        return fb_model
        
    except Exception as e:
        print('Training model error:', e)

def finetune_subword2vec(tokenized_traindata, inputbinpath, outputpath, vector_size=200, modelname="nberr_subword2vec_finetune.model"): # fsttext
    cores = multiprocessing.cpu_count()

    try:
        cap_path = datapath(inputbinpath)
        fb_model = gensim.models.fasttext.load_facebook_model(cap_path)
        
        fb_model.build_vocab(tokenized_traindata, update=True)
        fb_model.train(tokenized_traindata, total_examples=len(tokenized_traindata), epochs=fb_model.epochs)

        fb_model.save(str(outputpath.joinpath(modelname)))
        print('Training has finished. Model saved in file.')
        
        return fb_model
        
    except Exception as e:
        print('Training model error:', e)