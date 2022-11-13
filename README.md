# QConcursos-2022
https://github.com/brunoklaus/QConcursos-2022

Solution for 'The Adaptive Learning Challenge' proposed by QConcursos and hosted by SigmaGeek
---
Main file is ``QConcursos.ipynb``. It is assumed that the input csvs are put into the ``./data`` folder

There's not much to it, just a simple LSTM model in Pytorch. The way it is trained is noteworthy, however.  The ``__get__`` function for the training Dataset selects, for the given user, a random starting row ``beg`` from 0 to 99 as the starting point. The endpoint is set to ``end = beg + 50``; if this endpoint is greater than 99, then we wrap around so that the observed rows are ``[beg, .. 99, 0 , .. end-1]``. This variability of the training set seems to improve results.

<br/><br/>
The model is trained to predict whether the question at row ``end`` is answered correctly. It is trained with a binary cross-entropy loss function that incorporates appropriate weights to take the imbalance of ones versus zeroes into account. The input at each step of the LSTM are the embeddings for questions and columns. 
<br/><br/>
The best threshold for F1 score for the validation set is used when predicting the test set. In ``QConcursos.ipynb``, we load the pre-trained model (whose weights are given by ``pred.pt``) and ensure that the prediction is the same as the one sent to SigmaGeek.
