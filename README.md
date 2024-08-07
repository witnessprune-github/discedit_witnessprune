# Readme

This is the initial code for the submission: **DisCEdit: Model Editing by Identifying Discriminative Components**


### Requirements

1. PyTorch 2.01 and CUDA 11.7
2. CVXPY (any version should work)



### Organization

1. wtprn_X files are for DisCEdit-SP (to be renamed soon ) with different witness functions (can be a little slow for C100 models)
2. X_prune_cc files are for DisCEdit-U
3. cc_plots_X files are to generate discriminative filter plots (i.e. Figures 5/6 in the manuscript)
