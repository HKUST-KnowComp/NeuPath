# NeuPath
The codes for CIKM 2021 paper "Neural PathSim for Inductive Similarity Search in Heterogeneous Information Networks"
https://arxiv.org/pdf/2109.01549.pdf

Dependencies
------------
- python 3.7
- PyTorch 1.6+
- DGL 0.5.0+

Running
-------

Run with following to get the PathSim scores for the meta-path "PAFAP"

python train.py --n_layer 2 --n_head 4 --n_inp 256 --n_hid 256 ----gpu 0 

Miscellaneous
-------
Please send any questions you might have about the code and/or the algorithm to wxiaoae@cse.ust.hk.
