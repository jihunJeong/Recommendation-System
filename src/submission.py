import pandas as pd

answer_key1 = pd.read_table('./test/u1.base_prediction.txt', sep='\s+', names=['user','movie','rating'])
answer_key2 = pd.read_table('./test/u2.base_prediction.txt', sep='\s+', names=['user','movie','rating'])
answer_key3 = pd.read_table('./test/u3.base_prediction.txt', sep='\s+', names=['user','movie','rating'])
answer_key4 = pd.read_table('./test/u4.base_prediction.txt', sep='\s+', names=['user','movie','rating'])
answer_key5 = pd.read_table('./test/u5.base_prediction.txt', sep='\s+', names=['user','movie','rating'])

answer_key = pd.concat([answer_key1,answer_key2,answer_key3,answer_key4,answer_key5], ignore_index=True)
answer_key['id'] = answer_key.index
answer_key = answer_key[['id','rating']]
answer_key.to_csv('./test/submission.csv', sep=',', index=False)
print("Done")