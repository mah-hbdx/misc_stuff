#%%
# mamba install scvi-tools -c bioconda -c conda-forge
import hummingbird as hbdx
import numpy as np
import scvi
import scanpy as sc
import matplitlib.pyplot as plt
import pandas as pd

#%%
print(hbdx)
ad = hbdx.io.load('~/data/LC__ngs__rpm_log-21.5.0.h5ad')
#%%

print(ad.obs.head())

# %%


# %%
ad = hbdx.pipeline.FeatureThreshold(threshold=np.log2(1 + 100)).fit_transform(ad)
# %%
ad = ad.copy()
scvi.data.setup_anndata(ad)
# %%
model = scvi.model.LinearSCVI(ad, n_latent=10)

# %%
model.train(max_epochs=250, plan_kwargs={'lr':5e-3}, check_val_every_n_epoch=10, early_stopping=True)

# %%

train_elbo = model.history['elbo_train'][1:]
test_elbo = model.history['elbo_validation']

ax = train_elbo.plot()
test_elbo.plot(ax = ax)

# %%
