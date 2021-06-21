from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np
import sys
import os
import math
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import KFold

class RecDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = torch.from_numpy(dataset.values)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=20):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=True)
        self.movie_factors = nn.Embedding(n_movies, n_factors, sparse=True)

        self.user_biases = nn.Embedding(n_users, 1, sparse=True)
        self.movie_biases = nn.Embedding(n_movies, 1, sparse=True)

        self.dropout = nn.Dropout(0.5)

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_factors.weight, 0, 0.1)
        nn.init.normal_(self.movie_factors.weight, 0, 0.1)
        nn.init.normal_(self.user_biases.weight, 0, 0.1)
        nn.init.normal_(self.movie_biases.weight,0, 0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, user, movie):
        b = self.user_biases(user)
        b += self.movie_biases(movie)

        user_factor = self.user_factors(user)
        user_factor = self.dropout(user_factor)
        movie_factor = self.movie_factors(movie)
        movie_factor = self.dropout(movie_factor)
        
        x = ((user_factor * movie_factor).sum(dim=1, keepdim=True))
        x += b
        x = x.squeeze()
        return x

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    log_dir = './model'
    np.random.seed(42)

    print("Load Data ... ", end="", flush=True)
    df_ratings = pd.read_csv("./data-2/"+train, sep='\s+', names=['user','movie','rating','timestamp'])
    df_ratings.drop('timestamp', axis=1, inplace=True)
    
    ratings = pd.pivot_table(data=df_ratings, values='rating', index='user',columns='movie')
    dataset = RecDataset(df_ratings)

    users = df_ratings['user']
    movies = df_ratings['movie']

    user2idx = {user:idx for idx, user in enumerate(users.unique())}
    movie2idx = {movie:idx for idx, movie in enumerate(movies.unique())}
    n_users = len(df_ratings.user.unique())
    n_movies = len(df_ratings.movie.unique())
    print("Done")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    k_folds = 20
    set_fold = 5
    EPOCHS = 50
    kfold = KFold(n_splits=k_folds, shuffle=True)
    rmse = [0] * 5
    
    print("Matrix Factorization ... ", flush=True)
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        if fold == set_fold:
            break
        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(valid_ids)
        
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, pin_memory=True)
        valid_loader = DataLoader(dataset, batch_size=32, sampler=valid_sampler, pin_memory=True)
        
        best_rmse = 100

        model = MatrixFactorization(n_users, n_movies, n_factors=50).to(device)
        loss_fn = nn.MSELoss() 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(EPOCHS):
            train_loss = AverageMeter()
            valid_loss = AverageMeter()

            model.train()
            for iter, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                user = batch[:,0]
                movie = batch[:,1]
                rating = batch[:,2].to(torch.float32).to(device)

                user_ids = torch.LongTensor([user2idx[int(x)] for x in user]).to(device)
                movie_ids = torch.LongTensor([movie2idx[int(x)] for x in movie]).to(device)

                prediction = model(user_ids, movie_ids)
                loss = loss_fn(prediction, rating)
                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), len(batch))

                if (iter+1) % 100 == 0:
                    print("\rFolds [%d/%d] | Epochs [%d/%d] | Iter [%d/%d] | Train Loss %.4f" % (fold+1, set_fold, epoch+1, EPOCHS, iter+1, len(train_loader), train_loss.avg), end=" ")
            
            print("\rFolds [%d/%d] | Epochs [%d/%d] | Iter [%d/%d] | Train Loss %.4f" % (fold+1, set_fold, epoch+1, EPOCHS, iter+1, len(train_loader), train_loss.avg), end=" ")
            model.eval()
            for iter, batch in enumerate(valid_loader):
                user = batch[:,0]
                movie = batch[:,1]
                rating = batch[:,2].to(torch.float32).to(device)

                user_ids = torch.LongTensor([user2idx[int(x)] for x in user]).to(device)
                movie_ids = torch.LongTensor([movie2idx[int(x)] for x in movie]).to(device)

                with torch.no_grad():
                    prediction = model(user_ids, movie_ids)
                    vloss = loss_fn(prediction, rating)
                    valid_loss.update(vloss.item(), len(batch))

            print("Valid loss %.4f" % (valid_loss.avg))
            if best_rmse > valid_loss.avg:
                best_rmse = valid_loss.avg
                rmse[fold] = best_rmse
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)

                print(f"Model Save : rmse - {best_rmse}")
                torch.save(model, f'{log_dir}/{train[:2]}_model{fold+1}.pt')
    print("Done")

    print("Rating ... ", flush=True)
    df_test = pd.read_csv("./data-2/"+test, sep='\s+', names=['user','movie','rating','timestamp'])
    result_df = df_test.drop(['timestamp'], axis=1).copy()
    result_df.astype({'user':'int','movie':'int','rating':'float'})
    result = np.zeros((len(result_df), 5))
    for kfold in range(set_fold):
        print("\rFold [%d/%d]"% (kfold+1, set_fold), end="", flush=True)
        model = torch.load(log_dir+f"/{train[:2]}_model{kfold+1}.pt")
        model.to(device)
        model.eval()
        for idx, row in result_df.iterrows():
            user = row['user']
            movie = row['movie']
            if user not in users.unique() or movie not in movies.unique():
                key = ratings.loc[user].notnull()
                result[idx][kfold] = sum(ratings.loc[user, key]) / len(key)
                continue
        
            u = user2idx[user]
            m = movie2idx[movie]

            u = torch.LongTensor([int(u)]).to(device)
            m = torch.LongTensor([int(m)]).to(device)
            result[idx][kfold] = float(model(u, m))
    
    div = 0
    for i in range(set_fold):
        div += (1/rmse[i])
    for idx, row in result_df.iterrows():
        s = 0
        for i in range(set_fold):
            s += result[idx][i] / rmse[i]
        result_df.loc[idx, 'rating'] = s / div
    result_df.to_csv('./test/'+train[:2]+".base_prediction.txt", sep='\t', index=False, header=False)
    print("Done")