# timeseries
TCN architecture for time series

## Evaluation
### Methodology

Evaluation of tcn was done with respect to following points:
- Split selected dataset into training, evaluation and testing subsets
- Test set has lenght of `seq_length + n_ahead` where `seq_length` is length of sequence fed to TCN, and `n_ahead` is prediction length
- Train network using training and evaluation sets
- Feed `seq_lenght` of test set into trained network and get `n_ahead` predictions
- Evaluate `n_ahead` predictions against `n_ahead` from test set

## References
- [Bai et al., 2018, 
An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf)
