import pandas as pd
import numpy as np


class FileStreamer:
    def __init__(
        self, 
        source:str, 
        cols = None, 
        drop_cols = None,
        label_col = None,
        batch_size = 10):
        
        self.data = pd.read_csv(source)
        self.label=None
        if cols != None:
            self.data = self.data[cols]
            
        if drop_cols != None:
            self.data = self.data.drop(columns=drop_cols)
            
        if label_col!=None:
            self.label = self.data[label_col]
            self.data = self.data.drop(columns=label_col)
            
        self.index = 0
        self.batch_size=batch_size
        
    
    def __iter__(self):
        self.index = -1
        return self
    
    
    def __next__(self):
        if self.index < self.__len__():
            self.index += 1
            i = self.index * self.batch_size
            j = (self.index+1) * self.batch_size
            
            if self.label!=None:
                return self.data.iloc[i:j], self.label.iloc[i:j]
            return self.data.iloc[i:j]
        else:
            raise StopIteration
    
    
    def __len__(self):
        return len(self.data)//self.batch_size
    
    def reset(self):
        self.index = -1
        
        
class DynamicMinMax:
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = 1
        self.isfit=False
        
    def transform(self, X):
        if self.isfit:
            X_std =  (X - self.mins) / (self.maxs - self.mins)
            return X_std * (self.high - self.low) + self.low
        else:
            raise Exception("Model is not yet fit.")
            
            
    def fit(self, X):
        if self.isfit:
            mins = X.min().to_numpy()
            maxs = X.max().to_numpy()
            
            self.mins = np.concatenate(
                (mins.reshape(-1,1), self.mins.reshape(-1,1)), 
                axis=1).min(axis=1)
            
            self.maxs = np.concatenate(
                (maxs.reshape(-1,1), self.maxs.reshape(-1,1)), 
                axis=1).max(axis=1)
        else:
            self.mins = X.min().to_numpy()
            self.maxs = X.max().to_numpy()
            self.isfit=True
        return self