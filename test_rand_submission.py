import numpy as np
import pandas as pd

data = pd.read_csv('data/test.csv')
data_num,_ = data.shape

rand_probs = np.random.uniform(0,1,(data_num,6))

rand_total = np.sum(rand_probs,axis=1)
rand_probs = rand_probs/rand_total[:,None]

id = data['id']

pred = pd.DataFrame(data=rand_probs,
                   columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
pred.insert(0,'id',id)

pred.to_csv('data/submission1.csv')