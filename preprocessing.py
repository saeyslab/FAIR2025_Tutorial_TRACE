## Import libraries
import pandas as pd
import numpy as np
import readfcs
import os

## Load the data
def load_data(path, batch1='Gates_PTLG021_Unstim_Control_1', batch2='Gates_PTLG034_Unstim_Control_1'):

    batch1_unstim_fname = batch1
    batch2_unstim_fname = batch2

    batch1_unstim_data     = readfcs.read(os.path.join(path, batch1_unstim_fname + '.fcs'))
    batch2_unstim_data     = readfcs.read(os.path.join(path, batch2_unstim_fname + '.fcs'))
    batch1_unstim_labels   = pd.read_csv(os.path.join(path, batch1_unstim_fname + '.csv'), names=['cell_type'], header=0)
    batch2_unstim_labels   = pd.read_csv(os.path.join(path, batch2_unstim_fname + '.csv'), names=['cell_type'], header=0)

    ## Transform the data
    # Apply arcsinh transformation with cofactor 5
    dataset_list = [batch1_unstim_data, batch2_unstim_data]
    for i in range(len(dataset_list)):
        dataset_list[i].X = np.arcsinh(dataset_list[i].X / 5)

    ## Select the lineage and state markers for the analysis
    channels_of_interest = [48, 46, 43, 45, 20, 16, 21, 19, 22, 50, 47, 40, 44, 33, 17,
                            11, 18, 51, 14, 23, 32, 10, 49, 27, 24, 31, 42, 37, 39, 34,
                            41, 26, 30, 28, 29, 25, 35]
    other_channels = [9, 13, 15, 36, 38, 52]

    ## Create a combined dataset (combine the two batches)
    data = pd.concat([pd.DataFrame(batch1_unstim_data.X[:,channels_of_interest], columns=batch1_unstim_data.var.index[channels_of_interest]),
                    pd.DataFrame(batch2_unstim_data.X[:,channels_of_interest], columns=batch2_unstim_data.var.index[channels_of_interest])],
                    ignore_index=True)
    labels = pd.concat([batch1_unstim_labels, batch2_unstim_labels], ignore_index=True)
    batch = pd.DataFrame(['1']*len(batch1_unstim_data.X) + ['2']*len(batch2_unstim_data.X), columns=['batch'])
    state_markers = pd.concat([pd.DataFrame(batch1_unstim_data.X[:,other_channels], columns=batch1_unstim_data.var.index[other_channels]),
                            pd.DataFrame(batch2_unstim_data.X[:,other_channels], columns=batch2_unstim_data.var.index[other_channels])],
                            ignore_index=True)

    ## Save the data and metadata
    frames = [labels, batch, state_markers]
    metadata = pd.concat(frames, axis=1)
    metadata.to_csv(os.path.join(path, "ImmuneW_metadata.csv"), index=False)
    print('Metadata saved.')

    ### Save the data
    data.to_csv(os.path.join(path, "ImmuneW_HDdata.csv"), index=False)
    print('High-Dimensional data saved.')


