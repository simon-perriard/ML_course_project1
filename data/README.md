#Structure of the data

The 25th column (PRI_jet_num) seems to be linked to the test setup.
There are 4 different values (0 to 3) and they are linked to the -999 values in some columns.
It means that we have to train 4 different models to avoid to be impacted by the -999 values.
Our first task should be separating the data into 4 categories and train our models on these separated data.
