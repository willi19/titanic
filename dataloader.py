import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class PassengerDataSet(Dataset):
    def __init__(self, filename, train, validation = False):    
        df = pd.read_csv(filename)
        self.validation = validation
        self.train = train
        self.train_len = 800
        data = self.data_process(df)
        if train:
            if validation:
                self.target = torch.tensor(np.array(df["Survived"][self.train_len:]), dtype = torch.float32)
            else:
                self.target = torch.tensor(np.array(df["Survived"][:self.train_len]), dtype = torch.float32)
        self.data = torch.tensor(data, dtype = torch.float32)
        
        
    def normalize(self,df):
        return (df - df.mean())/df.std()

    #age age_nan sex sex_nan cabin cabin_nan  
    def data_process(self,df):
        data = []
        age_nan = df["Age"].isnull()
        age = self.normalize(df["Age"])
        if self.validation:
            for i in range(self.train_len, len(df)):
                personal_info = np.zeros(8)
                personal_info[0] = (0 if age_nan[i] else age[i])
                personal_info[1] = (1 if age_nan[i] else 0)
                personal_info[2] = (1 if df["Sex"].iloc[i] == "male" else 0)
                personal_info[3] = df["Parch"].iloc[i]
                personal_info[4] = df["SibSp"].iloc[i]
                personal_info[5] = (1 if df["Pclass"].iloc[i] == 1 else 0)
                personal_info[6] = (1 if df["Pclass"].iloc[i] == 2 else 0)
                personal_info[7] = (1 if df["Pclass"].iloc[i] == 3 else 0)
                data.append(personal_info)
        elif self.train:
            for i in range(self.train_len):
                personal_info = np.zeros(8)
                personal_info[0] = (0 if age_nan[i] else age[i])
                personal_info[1] = (1 if age_nan[i] else 0)
                personal_info[2] = (1 if df["Sex"].iloc[i] == "male" else 0)
                personal_info[3] = df["Parch"].iloc[i]
                personal_info[4] = df["SibSp"].iloc[i]
                personal_info[5] = (1 if df["Pclass"].iloc[i] == 1 else 0)
                personal_info[6] = (1 if df["Pclass"].iloc[i] == 2 else 0)
                personal_info[7] = (1 if df["Pclass"].iloc[i] == 3 else 0)
                data.append(personal_info)
        else:
            for i in range(len(df)):
                personal_info = np.zeros(8)
                personal_info[0] = (0 if age_nan[i] else age[i])
                personal_info[1] = (1 if age_nan[i] else 0)
                personal_info[2] = (1 if df["Sex"].iloc[i] == "male" else 0)
                personal_info[3] = df["Parch"].iloc[i]
                personal_info[4] = df["SibSp"].iloc[i]
                personal_info[5] = (1 if df["Pclass"].iloc[i] == 1 else 0)
                personal_info[6] = (1 if df["Pclass"].iloc[i] == 2 else 0)
                personal_info[7] = (1 if df["Pclass"].iloc[i] == 3 else 0)
                data.append(personal_info)
        data = np.array(data)
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if self.train:
            return self.data[idx], self.target[idx]
        else:
            return self.data[idx]

