import os
import pandas as pd
import numpy as np
import anndata as ad

def store_embeddings(embeddings, HD_data, metadata, fname):
    # Load the embeddings
    for emb in embeddings:
        filename = fname+f"_{emb}.csv"
        if os.path.exists(filename):
            if emb == 'umap':
                umap_emb = pd.read_csv(filename)
            elif emb == 'tsne':
                tsne_emb = pd.read_csv(filename)
            elif emb == 'pca':
                pca_emb = pd.read_csv(filename)
            elif emb == 'vivae':
                vivae_emb = pd.read_csv(filename)
            else:
                print(f"Unknown embedding type: {emb}")

    # Load metadata
    metadata = pd.read_csv(fname+"_metadata.csv")
    metadata_dict = {
        col: np.asarray(metadata[col].values, dtype=np.str_)
        if col == 'cell_type'
        else np.asarray(metadata[col].values, dtype=np.str_)
        if col == 'batch'
        else np.asarray(metadata[col].values, dtype=np.float32)
        for col in metadata.columns
    }

    # Load the data
    HD_data = pd.read_csv(fname+"_HDdata.csv")

    # Create an AnnData object
    d = ad.AnnData(
        X=HD_data.astype(np.float32),
        obs=metadata_dict,
        obsm={
            'tSNE': np.asarray(tsne_emb).astype(np.float32),
            'UMAP': np.asarray(umap_emb).astype(np.float32),
            'PCA': np.asarray(pca_emb).astype(np.float32),
            'ViVAE': np.asarray(vivae_emb).astype(np.float32)
        },
        uns={
            'methods': {
                'tSNE': ['tSNE'],
                'UMAP': ['UMAP'],
                'PCA': ['PCA'],
                'ViVAE': ['ViVAE']
            }
        }
    )

    # Save the AnnData object
    d.write(filename=fname + '_anndata.h5ad')
    print('Anndata object created successfully!')
    return d

def load_tracedata(fname, n_neighbours):
    if not os.path.exists(fname + '_TRACE.h5ad') and os.path.exists(fname + '_anndata.h5ad'):
        import sys
        sys.path.insert(1, '../TRACE-explainability/backend/')
        sys.path.insert(1, '../TRACE-explainability/')
        from dataset import Dataset as TraceData
        d = ad.read_h5ad(fname + '_anndata.h5ad')
        trace_data = TraceData(
            adata=d,
            name=fname + '_TRACE.h5ad',
            verbose=True,
            hd_metric="euclidean",
        )

        trace_data.precompute_HD_neighbors(maxK=1000)

        trace_data.compute_neighborhood_preservation(
            neighborhood_sizes=n_neighbours,
        )
        trace_data.compute_global_distance_correlation(
            max_landmarks=15000, LD_landmark_neighbors=True,
            hd_metric="euclidean", sampling_method="random",
        )
        trace_data.compute_random_triplet_accuracy(
            num_triplets=10
        )
        trace_data.compute_point_stability(num_samples=50)

        trace_data.align_embeddings(reference_embedding="PCA")
        trace_data.save_adata(filename=fname + '_TRACE.h5ad')

        trace_data.print_quality()
    else:
        d = ad.read_h5ad(fname + '_TRACE.h5ad')
    return d