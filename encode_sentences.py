from sentence_transformers import SentenceTransformer
import pickle, os
from tqdm import tqdm
import numpy as np
import re, bz2
import pandas as pd
import json
import time

DATA_DIR = '/pool001/jschless/quotebank/'
for year in range(2008, 2018, 1):
    # try processing data
    print('processing year', year)

    def enquote(s):
        return '"' + s + '"'

    if not os.path.exists(os.path.join(DATA_DIR, f'processed_quotes-{year}.csv')):
        print('processing quotes from compressed file')
        article_keys = ['articleID', 'url', 'date']
        quote_keys = ['quotation', 'globalTopSpeaker']
        f1 = os.path.join(DATA_DIR, f'quotebank-{year}.json.bz2')
        f2 = os.path.join(DATA_DIR, f'processed_quotes-{year}.csv')
        with bz2.open(f1) as f, open(f2, 'w') as w:
            print(','.join([*article_keys, *quote_keys]), file=w)
            for line in f:
                d = json.loads(line.decode())
                art_info = [d['articleID'], enquote(d['url']), d['date']]
                for q in d['quotations']:
                    print(','.join([*art_info, enquote(q['quotation']),
                    enquote(q['globalTopSpeaker'])]), file=w)
        df = pd.read_csv(os.path.join(DATA_DIR, f'processed_quotes-{year}.csv'), error_bad_lines=False)
    else:
        # csv has bugs in it
        print('processing quotes from already processed file')
        df = pd.read_csv(os.path.join(DATA_DIR, f'processed_quotes-{year}.csv'),
                         error_bad_lines=False, warn_bad_lines=False)
        df = df.reset_index()
        df.columns = ['articleID', 'url', 'date', 'quotation', 'globalTopSpeaker']

    if not os.path.exists(os.path.join(DATA_DIR, f'{year}_embeddings.npy')):
        print('generating embeddings')
        model = SentenceTransformer('stsb-roberta-base')

        # TODO use multiple GPUs

        print('starting encoding', time.asctime(time.localtime(time.time())))
        pool = model.start_multi_process_pool()
        #
        # #Compute the embeddings using the multi-process pool
        emb = model.encode_multi_process(df.quotation.to_list(), pool)
        del df
        X_sbert = np.vstack(corpus_embeddings).T
        del corpus_embeddings
        np.save(os.path.join(DATA_DIR, f'{year}_embeddings.npy'), X_sbert)

        print("Embeddings computed. Shape:", emb.shape)
        #
        # #Optional: Stop the proccesses in the pool
        model.stop_multi_process_pool(pool)
        print('finished encoding', time.asctime(time.localtime(time.time())))


        # corpus_embeddings = model.encode(df.quotation.to_list(), show_progress_bar=True)

    break
