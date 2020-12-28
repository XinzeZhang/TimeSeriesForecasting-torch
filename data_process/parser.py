import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse

def get_parser():
    """
    Generate a parameters parser.
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser(description="Time Series Forecasting with pytorch")

    # -----------------------------------------------------------------------------------
    # dataset location parameters
    parser.add_argument('-datafolder', type=str, default='paper.esm', help='folder name of the dataset')
    parser.add_argument('-dataset', type=str, default='ar1',help='file name of the dataset')
    
    # -----------------------------------------------------------------------------------
    # preprocess parameters
    parser.add_argument('-normal', action='store_true',
                        help='Whether to use standard scaler')
    parser.add_argument('-diff', action='store_true',
                        help='Whether to diff')

    # -----------------------------------------------------------------------------------
    # forecasting horizon parameters
    parser.add_argument('-H', type=int, default=1, metavar='N',
                        help='steps for prediction (default: 1)')

    # -----------------------------------------------------------------------------------
    # model parameters

    ## share parameters
    parser.add_argument('-restore', action='store_true',
                        help='Whether to restore the model state from the best.pth.tar')
    
    ## parameters of the training model
    parser.add_argument('-num_epochs', type=int, default=1000, metavar='N',
                        help='epochs for training')
    parser.add_argument('-k', type=int, default=5,help='k-fold for cross-validation')

    ### parameters of deepAR model
    parser.add_argument('-sample-dense', action='store_true', default=True,
                    help='Whether to continually sample the time series during preprocessing')
    parser.add_argument('--relative-metrics', action='store_true',
                    help='Whether to normalize the metrics by label scales')
    parser.add_argument('--sampling', action='store_true',
                    help='Whether to sample during evaluation')
    parser.add_argument('--save-best', action='store_true',
                    help='Whether to save best ND to param_search.txt')

    ## parameters of the random model
    ### parameters of the ES/ESM model
    parser.add_argument('-search', type=str, default='random', help='method of generating candidates')

    # -----------------------------------------------------------------------------------
    # experimental log parameters
    parser.add_argument('-test', action='store_true',
                        help='Whether to test')

    # -----------------------------------------------------------------------------------
    # experiment repetitive times
    parser.add_argument('-times', type=int, default=20, help='experiment repetitive times')

    params = parser.parse_args()

    return params

if __name__ == "__main__":
    opts = get_parser()