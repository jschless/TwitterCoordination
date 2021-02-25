
# coding: utf-8

# In[5]:

import preprocessing
from sentence_transformers import SentenceTransformer


# In[6]:

campaigns = preprocessing.load_campaign()
users = preprocessing.load_user_data()


# In[7]:

all_cascades = preprocessing.process_campaign(campaigns)


# In[8]:

roots = [x.root for x in all_cascades]


# # Load Pre-Trained Model

# In[9]:

model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')


# In[12]:

from config import TWITTER_DATA_DIR


# In[16]:

import os, pickle


# In[17]:

with open(os.path.join(TWITTER_DATA_DIR, 'cascade_root_ids_2.pkl'), 'wb') as f:
    pickle.dump([x.id for x in roots], f)
    print('dumped list of root ids')


# In[21]:

roots[0].text


# In[ ]:
print('starting encoding')

corpus_embeddings = model.encode([x.text for x in roots], show_progress_bar=True)


# In[ ]:

with open(os.path.join(TWITTER_DATA_DIR, 'cascade_root_embeddings_2.pkl'), 'wb') as f:
    pickle.dump(corpus_embeddings, f)


# ### Future Trial Removing Hashtags
# Really should remove non text things

# In[ ]:



