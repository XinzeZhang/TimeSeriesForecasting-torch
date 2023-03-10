# Torch-Forecasting

This is the repository of time series forecasting models modified by Xinze Zhang.

Based on the work of other researchers, this repository is expected to make the codes easy-reading as well as providing more succinct, standard, easy-to-use API that can forecast time series.

## Note

The selected dataset and the parameters of the selected models are defined in the `exp/**` folder. 
E.g., the experiment on the PM2.5 dataset with ESN model can be conducted by:
```
python exp/eto/pm.py -model esn
```

Due to the space limitation of the Github and the huge size of the checking_points as well as the logs, we only upload the datasets and model files related to the our papers.

For the experimental logs, pre-trained model checking_points, and results files, contact the code [contributor](xinze@hust.edu.cn) of this repository.

### Related Works
#### DeepAR

`DeepAR` ([Paper](https://arxiv.org/abs/1704.04110)) is a pytorch-based modification of the implementation in [TimeSeries](https://github.com/zhykoties/TimeSeries).

#### CLSTM

`CLSTM` ([Paper](https://arxiv.org/abs/1903.02540))is a pytorch-based modification of the implementation in [ConvRNN](https://github.com/KurochkinAlexey/ConvRNN).

#### MSVR

`MSVR` ([Paper](https://www.sciencedirect.com/science/article/abs/pii/S092523121300917X)) is a numpy-based implementation by our work in [MSVR](https://github.com/Analytics-for-Forecasting/msvr).

## Acknowledgement

- This work was done under the direction of our supervisor Prof. Yukun Bao.
