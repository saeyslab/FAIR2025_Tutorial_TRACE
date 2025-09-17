## Import libraries
import pandas as pd
import numpy as np
import readfcs
import os

## Load the data
def load_data(path, batch1='Gates_PTLG021_Unstim_Control_1', batch2='Gates_PTLG021_IFNa_LPS_Control_1'):

    unstim_fname = batch1
    ifna_lps_fname = batch2

    unstim_data     = readfcs.read(os.path.join(path, unstim_fname + '.fcs'))
    ifna_lps_data   = readfcs.read(os.path.join(path, ifna_lps_fname + '.fcs'))
    unstim_gated    = pd.read_csv(os.path.join(path, unstim_fname + '_gated.csv'))
    ifna_lps_gated  = pd.read_csv(os.path.join(path, ifna_lps_fname + '_gated.csv'))

    ## Transform the data
    # Apply arcsinh transformation with cofactor 5
    dataset_list = [unstim_data, ifna_lps_data]
    for i in range(len(dataset_list)):
        dataset_list[i].X = np.arcsinh(dataset_list[i].X / 5)

    ## Select the lineage and state markers for the analysis
    ## Select the channels of interest
    channels_of_interest = [47, 45, 42, 44, 19, 15, 20, 18, 21, 49, 46, 39, 43, 32,
                            10, 17, 50, 13, 22, 31,  9, 48, 26, 23, 30, 41, 36, 38, 33,
                            40, 25, 29, 27, 28, 24, 34]
    other_channels = [16, 51, 52] 

    ## Filter the gated data
    gate_CD235_CD61_unstim = unstim_gated['CD235.CD61.']
    gate_CD235_CD61_ifna_lps = ifna_lps_gated['CD235.CD61.']

    filtered_unstim_data = pd.DataFrame(unstim_data.X[np.ix_(gate_CD235_CD61_unstim, channels_of_interest)], columns=unstim_data.var.index[channels_of_interest])
    filtered_ifna_lps_data = pd.DataFrame(ifna_lps_data.X[np.ix_(gate_CD235_CD61_ifna_lps, channels_of_interest)], columns=ifna_lps_data.var.index[channels_of_interest])

    ## Create a combined dataset (combine the two batches)
    data = pd.concat([filtered_unstim_data, filtered_ifna_lps_data], ignore_index=True)
    labels = pd.concat([pd.DataFrame(unstim_gated['cell_type'][gate_CD235_CD61_unstim], columns=['cell_type']),
                    pd.DataFrame(ifna_lps_gated['cell_type'][gate_CD235_CD61_ifna_lps], columns=['cell_type'])],
                    ignore_index=True)
    batch = pd.DataFrame(['Unstimulated']*len(filtered_unstim_data) + ['IFN-α + LPS']*len(filtered_ifna_lps_data), columns=['batch'])
    state_markers = pd.concat([pd.DataFrame(unstim_data.X[np.ix_(gate_CD235_CD61_unstim, other_channels)], columns=unstim_data.var.index[other_channels]),
                           pd.DataFrame(ifna_lps_data.X[np.ix_(gate_CD235_CD61_ifna_lps, other_channels)], columns=ifna_lps_data.var.index[other_channels])],
                            ignore_index=True)

    ## Save the data and metadata
    metadata = pd.concat([labels, batch, state_markers], axis=1)
    metadata.to_csv(os.path.join(path, "ImmuneW_metadata.csv"), index=False)
    print('Metadata saved.')

    ### Save the data
    data.to_csv(os.path.join(path, "ImmuneW_HDdata.csv"), index=False)
    print('High-Dimensional data saved.')


