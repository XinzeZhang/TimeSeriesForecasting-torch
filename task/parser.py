import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse



def get_parser(parsing = True):
    """
    Generate a parameters parser.
    """
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser(description="Time Series Forecasting with pytorch")

    # -----------------------------------------------------------------------------------
    # Model name
    parser.add_argument('-model', type=str, default='mlp', help='name of the implemented model')


    # -----------------------------------------------------------------------------------
    # dataset location parameters
    parser.add_argument('-datafolder', type=str, default='paper.esm', help='folder name of the dataset')
    parser.add_argument('-dataset', type=str, default='Brent-d',help='file name of the dataset')
    
    # -----------------------------------------------------------------------------------
    # preprocess parameters
    parser.add_argument('-normal',default=False, action='store_true',
                        help='Whether to use standard scaler')
    
    parser.add_argument('-diff', action='store_true',
                        help='Whether to diff')

    # -----------------------------------------------------------------------------------
    # forecasting horizon parameters
    parser.add_argument('-H', type=int, default=1, metavar='N',
                        help='steps for prediction (default: 1)')
    # Multi-Output strategy  parameters
    parser.add_argument('-exp_name', type=str, default='mimo', metavar='N',
                        help='exp_name  (default: mimo)')

    # -----------------------------------------------------------------------------------
    # model parameters

    ## share parameters
    parser.add_argument('-restore', action='store_true',
                        help='Whether to restore the model state from the best.pth.tar')
    
    ## parameters of the training model
    parser.add_argument('-epochs', type=int, default=100, metavar='N',
                        help='epochs for training')
    parser.add_argument('-k', type=int, default=5,help='k-fold for cross-validation')

    ### parameters of cnn model
    parser.add_argument('-kernel_size', type=int, default=3, metavar='N',
                    help='kernel_size of the cnn model for prediction (default: 1)')

    ### parameters of deepAR model
    parser.add_argument('-sample-dense', action='store_true', default=True,
                    help='Whether to continually sample the time series during preprocessing')
    parser.add_argument('-relative-metrics', action='store_true',
                    help='Whether to normalize the metrics by label scales')
    parser.add_argument('-sampling', action='store_true',
                    help='Whether to sample during evaluation')
    parser.add_argument('-save-best', action='store_true',
                    help='Whether to save best ND to param_search.txt')

    ## parameters of the random model
    ### parameters of the ES/ESM model
    parser.add_argument('-search', type=str, default='random', help='method of generating candidates')

    # -----------------------------------------------------------------------------------
    # experimental log parameters
    parser.add_argument('-test', default=False,action='store_true',
                        help='Whether to test')
    parser.add_argument('-clean', default=True, action='store_true',
                        help='Whether to test')    
    parser.add_argument('-logger_level', type=int, default=20, help='experiment log level')

    # -----------------------------------------------------------------------------------
    # experiment repetitive times
    parser.add_argument('-rep_times', type=int, default=1, help='experiment repetitive times')

    parser.add_argument('-cuda',default=True,action='store_true', help='experiment with cuda')
    parser.add_argument('-gid',type=int, default=0, help='default gpu id')

    # -----------------------------------------------------------------------------------
    # tune resource parameters
    parser.add_argument('-tune', default=False, help='execute tune or not')
    # parser.add_argument('-cores', type=int, default=2, help='cpu cores per trial in tune')
    # parser.add_argument('-cards',type=int, default=0.25, help='gpu cards per trial in tune')
    # parser.add_argument('-tuner_iters', type=int, default=50, help='hyper-parameter search times')
    # parser.add_argument('-tuner_epochPerIter',type=int,default=1)


    parser.add_argument('-tag',type=str, default='', help='additional experimental model tag')
    # -----------------------------------------------------------------------------------
    # eval parameters
    parser.add_argument('-metrics', nargs='+', default=['rmse','mape','smape'], help='measures list')
    
    parser.add_argument('-sid', nargs='+', default=['all'], help='experimental series id')
    parser.add_argument('-cid', nargs='+', default=['all'], help='experimental cross validation id')
    
    
    if parsing:
        params = parser.parse_args()
        return params
    else:
        return parser
    # params, unknown = parser.parse_known_args()


if __name__ == "__main__":
    opts = get_parser()