# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 07:52:03 2020

@author: Ganesh
"""

import numpy as np
import pandas as pd
import tensorflow as tf


class Word2vec:
    # Input Parameters    
    def __init__(self,Corpus):
        self.Corpus=Corpus      
    # Create the Repetitive words as a list 
    def Unwanted_List(self): 
        self.Unwanted_Words=['The',"in","for","of","to","from","is","was"] # Defining Unwanted Words
        self.Unwanted_Words=[i.lower() for i in self.Unwanted_Words ]
    
    # Cleaning Input Corpus    
    def Clean_Input_Corpus(self):
        self.Corpus_Sent=[]
        self.Corpus_cln=self.Corpus.lower().strip().split()    # Converting to lower case
        self.Corpus_cln=[w.strip(".") for w in self.Corpus_cln] # Removing the punctuations
        self.Corpus_RawSent=self.Corpus.lower().split('.')         #Chunking into Sentences
        [self.Corpus_Sent.append(w.split()) for w in self.Corpus_RawSent] # List of List of Words in Sentences
        self.Corpus_Sent=list(filter(None, self.Corpus_Sent)) # Removing empty from above sentences
        self.Corpus_cln=list(set(self.Corpus_cln)) # Getting Unique words from main Corpus
        self.Voc=[w for w in self.Corpus_cln if w not in self.Unwanted_Words] # Defining the vocabulaory without Unwanted Words
        #print(self.Corpus_Sent1)
        #print(self.Corpus)
        #print(self.Voc)
    # Converting Words in Vocabulary to numbers and viceversa        
    
    def Word_To_Numeric(self):
        self.voc2int={} # Defining empty dict (keys: Words; Values: Numbers)
        self.int2voc={} # Defining empty dict (keys: Numbers; Values: Words)   
        for j,word in enumerate(self.Voc):
            self.voc2int[word]=j
            self.int2voc[j]=word    
        #print(self.voc2int)
        #print(self.int2voc)
        
    def Train_Data_Prep(self,word_window=2):
        #print(word_window)
        self.sample_data=[] # Final Array of the train data
        for x in range(len(self.Corpus_Sent)): # Loop through external sentence loop
            lenx=len(self.Corpus_Sent[x]) # Length inside each sentence loop
            for ind in range(lenx): 
            #for ind in range(1):
                # Training examples generation for forward window (+ Forward Window)
                try:
                    for word in range(1,word_window,1):
                        fir_list=[] # List which will be updated everytime with the x,y pairs
                        fir_list.append(self.Corpus_Sent[x][ind]) # x value of the pair
                        fir_list.append(self.Corpus_Sent[x][ind+word]) # Y value of the pair
                        self.sample_data.append(fir_list)                
                except:
                    pass # Since in the forward loop there is an error possibility ..ignore
                
                
        
                # Training examples generation for backward window (- Backward Window)
                try:
                    neg_list=[]
                    for word in range(-(word_window),0,1):
                        fir_list=[]
                        
                        if (ind+word) < 0 :
                            pass
                        else:                    
                            #print(ind,ind+word,sentences[x][ind],sentences[x][ind+word])
                            fir_list.append(self.Corpus_Sent[x][ind])
                            fir_list.append(self.Corpus_Sent[x][ind+word])
                            self.sample_data.append(fir_list)
                except:
                    pass # Since in the backward loop there is an error possibility ...ignore
        #print(self.sample_data)
        s
    def voc2onehot_fn(self,word,Voc1,voc_size):
        voc_size=len(Voc1) # length of the vocabulary
        word_onehot=np.zeros(voc_size) # One hot for each word with vocabulary length
        word_onehot[self.voc2int[word]]=1
        return word_onehot
    
        #word_onehot=voc2onehot_fn("work",Voc,13)
        #print(word_onehot)
        
    #print(len(sample_data))
    def train_data_act(self):
        self.x_train=[]
        self.y_train=[]
        for i in range(len(self.sample_data)):
            #print(sample_data[i][0].lower(),type(sample_data[i][0].lower()))
            #print(Unwanted_Words,type(Unwanted_Words[0]))
            if str(self.sample_data[i][0].lower()) in self.Unwanted_Words or str(self.sample_data[i][1].lower()) in self.Unwanted_Words:
                pass
            else:
                #print(self.sample_data[i][0],self.sample_data[i][1])                
                self.x_train.append(self.sample_data[i][0]) # X train value append
                self.y_train.append(self.sample_data[i][1]) # Y train value append
        #[x_train.append(sample_data[i][0] for i in range(len(sample_data)))]
        
        #print(x_train)
        #print(y_train)
        
        
        self.x_train_onehot=[]
        self.y_train_onehot=[]
        self.voc_size=len(self.Voc)
        for val in self.x_train:
            self.x_train_onehot.append(list(self.voc2onehot_fn(val,self.Voc,self.voc_size)))# Entire x_train One hot
        for val in self.y_train:
            self.y_train_onehot.append(list(self.voc2onehot_fn(val,self.Voc,self.voc_size)))# Entire y_train One hot
            
        self.x_train_onehot=np.float32(np.asarray(self.x_train_onehot))
        self.y_train_onehot=np.float32(np.asarray(self.y_train_onehot))

    def tf_Graph(self,EMBEDDING_DIM=5):
        # making placeholders for x_train and y_train
        self.x = tf.placeholder(tf.float32, shape=(None, self.voc_size))
        self.y_label = tf.placeholder(tf.float32, shape=(None, self.voc_size))
        
        EMBEDDING_DIM = 5 # you can choose your own number
        self.W1 = tf.Variable(tf.random_normal([self.voc_size, EMBEDDING_DIM]))
        self.b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
        self.hidden_representation = tf.add(tf.matmul(self.x,self.W1), self.b1)
        
        self.W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, self.voc_size]))
        self.b2 = tf.Variable(tf.random_normal([self.voc_size]))
        self.prediction = tf.add( tf.matmul(self.hidden_representation, self.W2), self.b2)       
                        
    def tf_Sess(self,epochs=10000):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        #sess.run(init) #make sure you do this!
        # define the loss function:
        cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_label,logits=self.prediction))
        #cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        # define the training step:
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
                # train for n_iter iterations
        print("Training in Progress")
        with tf.Session() as sess:
            sess.run(init)    
            for epoch in range(epochs):
                sess.run(train_step, feed_dict={self.x: self.x_train_onehot, self.y_label: self.y_train_onehot})
                #print('For the epoch-'+str(epoch)+' loss is : ', sess.run(cross_entropy_loss, feed_dict={self.x: self.x_train_onehot, self.y_label: self.y_train_onehot}))
            self.embed_vectors=sess.run(self.W1+self.b1)
        
        #print(self.voc_size,self.embed_vectors)
        
    def eucl_distance(self,vec1,vec2):
        return np.sqrt(np.sum(vec1-vec2)**2)
    
    def cos_sim(self,vec1,vec2):
        Nr=np.dot(vec1,vec2)
        Dr1=np.linalg.norm(vec1)
        Dr2=np.linalg.norm(vec2)
        return Nr/Dr1/Dr2
    
    def find_closest(self,test_word,mode="Euc"):
        min_dist=100000
        min_ind=-1
        
        max_cos=0
        max_ind_cos=-1
        
        vec_find=self.embed_vectors[self.voc2int[test_word]]
        
        for ind,val in enumerate(self.embed_vectors):
            if mode == "Euc": 
                if self.eucl_distance(val,vec_find) < min_dist and not np.array_equal(val,vec_find):
                    min_dist=self.eucl_distance(val,vec_find)
                    min_ind=ind
                    word_ind=min_ind
            else:
                if self.cos_sim(val,vec_find) > max_cos and not np.array_equal(val,vec_find):
                    max_cos=self.cos_sim(val,vec_find)
                    max_ind_cos=ind
                    word_ind=max_ind_cos
        print("Matching word for Input Word: "+str(test_word)+" is "+str(self.int2voc[word_ind]))            
        #return min_ind,max_ind_cos
                
        
        
        
Inp_Text="He is Ganesh. Ganesh is king of    the    jungle. Ganesh has a beautiful garden. He has a big house. He always sleeps in the day. She is Praarthana. Praarthana is Queen of the Jungle. She works all time of the day. She has a small house. Praarthana has a small house     "
#Inp_Text="Ganesh has a big house. Praarthana has a small house."

#Corpus,Voc=Clean_Input_Corpus(Inp_Text)
Inp1=Word2vec(Inp_Text)
Inp1.Unwanted_List()
Inp1.Clean_Input_Corpus()
Inp1.Word_To_Numeric()
Inp1.Train_Data_Prep(word_window=2)
Inp1.train_data_act()
Inp1.tf_Graph()
Inp1.tf_Sess()
Inp1.find_closest("beautiful",mode="cos")
#Unwanted_Dict()
    