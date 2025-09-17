# FAIR2025_Tutorial_TRACE

This repository provides tools to preprocess mass cytometry (CyTOF) data and compute lower-dimensional embeddings for visualization and analysis with [TRACE](https://github.com/aida-ugent/TRACE).

### Data availability
The data set used in this example is a mass cytometry data set immunoprofiling women during pregnancy [[Van GAssen, S. et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7078957/)]. The dataset used for this demonstration contains whole blood samples from two healthy volunteers, both unstimulated and stimulated with Interferonα (IFNα) and Lipopolysaccharide (LPS). The data can be downloaded from the *data* folder in this repository.
    
### Preprocessing
  - Data cleaning and transformation
  - Marker selection and filtering

### Dimensionality reduction
  - PCA
  - UMAP (n_neighbours=100)
  - tSNE (perplexity=100)
  - ViVAE (n_epochs=100, lam_mds=10)
