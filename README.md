# timeseries
TCN architecture for time series

## Evaluation

Temporal convolutional network was evaluated with respect to following points:
- Split selected dataset into training, evaluation and testing subsets
- Test set has lenght of `seq_length + n_ahead` where `seq_length` is length of sequence fed to TCN, and `n_ahead` is prediction length
- Train network using training and evaluation sets
- Feed `seq_lenght` of test set into trained network and get `n_ahead` predictions
- Evaluate `n_ahead` predictions against `n_ahead` from test set

### Evaluation set 1 - Historical Hourly Weather Data
[Source](https://www.kaggle.com/selfishgene/historical-hourly-weather-data)

<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/loss.png">
  </p>

<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/unnamed-chunk-5-1.png">
  </p>
  
<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/montreal_weather_model.png">
  </p>

### Evaluation set 2 - Air Quality in Madrid
[Source](https://www.kaggle.com/decide-soluciones/air-quality-madrid/home)

<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/loss2.png">
  </p>

<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/unnamed-chunk-9-1.png">
  </p>
  
<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/montreal_weather_model.png">
  </p>
  
### Evaluation set 3 - Hourly Energy Consumption
[Source](https://www.kaggle.com/robikscube/hourly-energy-consumption/home)

<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/unnamed-chunk-11-1.png">
  </p>

<p align="center">
  <img src="https://github.com/xadrianzetx/timeseries/blob/master/plots/unnamed-chunk-13-1.png">
  </p>

## TODO
- compare TCN to other models (LSTM, Holt Winters)

## References
- [Bai et al., 2018, 
An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/pdf/1803.01271.pdf)
