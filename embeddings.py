# This script contains functions to compute various embeddings (UMAP, t-SNE, PCA, Diffusion Map, PHATE, ViVAE) for a given dataset.

# Import libraries
import pandas as pd
import numpy as np
import umap
import openTSNE
from sklearn.decomposition import PCA
import os
import vivae as vv

# Set the random seed for reproducibility
random_state = 42

# Compute the UMAP embedding
def compute_umap(data, labels=None, n_components=2, n_neighbors=100):
    print('Computing the UMAP embedding...')
    model_umap = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    umap_emb = model_umap.fit_transform(data)
    if labels is None:
        umap_emb_df = pd.DataFrame(umap_emb, columns=['umap1', 'umap2'])
    else:
        umap_emb_df = pd.DataFrame(np.concatenate((umap_emb,labels),axis=1), columns=['umap1', 'umap2','labels'])
    return umap_emb_df

# Compute the t-SNE embedding
def compute_tsne(data, labels=None, n_components=2, perplexity=100):
    print('Computing the t-SNE embedding...')
    aff = openTSNE.affinity.PerplexityBasedNN(data, perplexity=perplexity, n_jobs=8, random_state=random_state)
    if os.path.exists('mouse_pca.csv'):
        pca = pd.read_csv('mouse_pca.csv')
        pca_emb = pca[['pc1', 'pc2']].values
        init = openTSNE.initialization.rescale(pca_emb)
        tsne_emb = openTSNE.TSNE(n_jobs=32,verbose=True).fit(affinities=aff, initialization=init)
    else:
        tsne_emb = openTSNE.TSNE(n_jobs=32,verbose=True).fit(affinities=aff)
    if labels is None:
        tsne_emb_df = pd.DataFrame(tsne_emb, columns=['tsne1', 'tsne2'])
    else:
        tsne_emb_df = pd.DataFrame(np.concatenate((tsne_emb,labels),axis=1), columns=['tsne1', 'tsne2','labels'])
    return tsne_emb_df
    
# Compute the PCA embedding    
def compute_pca(data, labels=None, n_components=2):
    print('Computing the PCA embedding...')
    pca = PCA(n_components=n_components)
    pca_emb = pca.fit_transform(np.array(data))
    if labels is None:
        pca_emb_df = pd.DataFrame(pca_emb, columns=['pc1', 'pc2'])
    else:
        pca_emb_df = pd.DataFrame(np.concatenate((pca_emb,labels),axis=1), columns=['pc1', 'pc2','labels'])
    return pca_emb_df

# Compute the ViVAE embedding
def compute_vivae(data, labels=None, n_components=2):
    print('Computing the ViVAE embedding...')
    model_vivae = vv.ViVAE(input_dim=data.shape[1], latent_dim=n_components)
    model_vivae.fit(data, n_epochs=100, lam_mds=10.)
    vivae_emb = model_vivae.transform(data)
    if labels is None:
        vivae_emb_df = pd.DataFrame(vivae_emb, columns=['vivae1', 'vivae2'])
    else:
        vivae_emb_df = pd.DataFrame(np.concatenate((vivae_emb,labels),axis=1), columns=['vivae1', 'vivae2','labels'])
    return vivae_emb_df

## Get the embeddings of the data
def compute_embeddings(emb_names: list, filename: str, hd_data: pd.DataFrame, labels: pd.DataFrame = None, n_components: int = 2, perplexity: int = 100, n_neighbors: int = 100):
    for emb in emb_names:
        ## PCA
        if emb == 'pca' and not os.path.exists(filename+"_pca.csv"):
            pca_emb = compute_pca(hd_data.values)
            pca_emb.to_csv(filename+"_pca.csv", index=False) 
            print('PCA embedding saved.')
        ## UMAP
        if emb == 'umap' and not os.path.exists(filename+"_umap.csv"):
            umap_emb = compute_umap(hd_data.values)
            umap_emb.to_csv(filename+"_umap.csv", index=False)
            print('UMAP embedding saved.')
        ## t-SNE
        if emb == 'tsne' and not os.path.exists(filename+"_tsne.csv"):
            tsne_emb = compute_tsne(hd_data.values)
            tsne_emb.to_csv(filename+"_tsne.csv", index=False)
            print('t-SNE embedding saved.')
        ## ViVAE
        if emb == 'vivae' and not os.path.exists(filename+"_vivae.csv"):
            vivae_emb = compute_vivae(hd_data.values)
            vivae_emb.to_csv(filename+"_vivae.csv", index=False)
            print('ViVAE embedding saved.')