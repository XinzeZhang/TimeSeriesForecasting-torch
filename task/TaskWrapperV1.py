import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
import math
from tqdm.std import tqdm
import importlib
from task.util import plot_xfit, plot_hError, scaler_inverse, IQR_check
from task.util import os_makedirs, os_rmdirs, set_logger
from task.TaskTuner import StocHyperTuner as HyperTuner
from task.TaskLoader import Opt
import torch
import numpy as np
import statistics
from tqdm import trange
import shutil
import copy
from models.statistical.naive import Naive
from collections.abc import Mapping
# from numpy.lib.function_base import select
# from ray.tune import logger
# from task.metric import rmse, mape, smape


class Task(Opt):
    def __init__(self, args):
        # self.opts = Opt()
        # self.opts.merge(args)

        self.exp_module_path = importlib.import_module('{}.{}'.format(
            args.datafolder.replace('/', '.'), args.dataset))
        
        self.data_config(args)
        self.model_config(args)
        self.exp_config(args)
        # self.data_subconfig()

    def data_config(self, args):
        self.data_name = args.dataset
        # data_opts = getattr(self.exp_module_path, args.dataset + '_data')
        data_opts = getattr(self.exp_module_path, 'Data')
        self.data_opts = data_opts(args)
        # loading args.H to each subPack

    def model_config(self, args):
        self.model_name = args.model

        # load the specifical config firstly, if not exists, load the common config
        if hasattr(self.exp_module_path, self.model_name):
            model_opts = getattr(self.exp_module_path,
                                 args.model)
        else:
            try:
                share_module_path = importlib.import_module('data.base')
                model_opts = getattr(
                    share_module_path, self.model_name + '_default')
            except:
                raise ValueError(
                    'Non-supported model "{}" in the "{}" module, please check the module or the model name'.format(self.model_name, self.exp_module_path))

        self.model_opts = model_opts()
        self.model_opts.hyper.merge(opts=self.data_opts.info)
        # self.model_opts.hyper.H = args.H # hyper H should be defined within the data subPack
        
        if 'hyper' in vars(args):
            self.model_opts.hyper.update(args.hyper)
            

    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        # model = model(self.model_opts.hyper, self.logger)  # transfer model in the conduct func.
        return model

    def exp_config(self, args, fit = True):
        cuda_exist = torch.cuda.is_available()
        if cuda_exist and args.cuda:
            self.device = torch.device('cuda:{}'.format(args.gid))
        else:
            self.device = torch.device('cpu')

        # if 'statistic' in vars(self.model_opts):
        # if fit:
        if fit and self.model_opts.arch == 'statistic':
            self.device = torch.device('cpu')

        self.exp_dir = 'trial' if args.test == False else 'test'

        if args.exp_name is not None:
            self.exp_dir = os.path.join(self.exp_dir, args.exp_name)

        self.exp_dir = os.path.join(
            self.exp_dir, 'normal') if self.data_opts.info.normal else os.path.join(self.exp_dir, 'minmax')

        assert args.diff == False  # Not support differential preprocess yet!
        # self.exp_dir += '_diff'

        self.exp_dir = os.path.join(self.exp_dir, args.dataset)
        self.fit_dir = os.path.join(self.exp_dir, 'fit')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')

        self.model_name = '{}'.format(args.model) if args.tag == '' else '{}_{}'.format(args.model, args.tag)

        if args.test and args.logger_level != 20:
            self.logger_level = 50  # equal to critical
        else:
            self.logger_level = 20  # equal to info

        self.rep_times = args.rep_times

        self.sid_list = args.sid
        self.cid_list = args.cid
        if self.sid_list == ['all']:
            self.sid_list = list(range((self.data_opts.info.num_series)))
        else:
            _temp = [int(s) for s in self.sid_list]
            self.sid_list = _temp
            
        if self.cid_list == ['all']:
            self.cid_list = list(range(self.rep_times))
        else:
            _temp = [int(c) for c in self.cid_list]
            self.cid_list = _temp

        # if args.test and args.clean:
        #     os_rmdirs(self.task_dir)
        # os_makedirs(self.task_dir)
        self.model_fit_dir_list = []
        self.model_eval_dir_list = []
        
        for sid in range(len(self.data_opts.seriesPack)):
            # H in this place should be binging with the corr. subPack.
            _H = self.data_opts.seriesPack[sid].H
            _task_dir = os.path.join(self.fit_dir, 'series{}'.format(sid), 'h{}'.format(_H), self.model_name)
            _eval_dir = os.path.join(
                    self.eval_dir, 'series{}'.format(sid), 'h{}'.format(_H), self.model_name)
            if args.test and args.clean:
                os_rmdirs(_task_dir)
            # os_makedirs(_task_dir)
            self.model_fit_dir_list.append(_task_dir)
            self.model_eval_dir_list.append(_eval_dir)
        

        if fit:
            self.model_opts.hyper.device = self.device
            self.tune = args.tune  # default False

    def logger_config(self, dir, stage, cv, sub_count):
        log_path = os.path.join(dir, 'logs',
                                '{}.cv{}.series{}.log'.format(stage, cv, sub_count))
        log_name = '{}.series{}.cv{}.{}'.format(
            self.data_name, sub_count, cv, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger

    def conduct(self,):
        # init and mkdir taskdir
        # generate the subPack dataset
        # if self.tune:
        #     self.tuning()

        # for sub_count, subPack in enumerate(tqdm()):
        for subPack in tqdm([self.data_opts.seriesPack[s] for s in self.sid_list]):

            # self.task_dir = os.path.join(self.task_dir, 'series_{}'.format(sub_count))
            sub_count = subPack.index
            self.model_fit_dir = self.model_fit_dir_list[sub_count]
            
            self.pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
            
            
            os_makedirs(self.pred_dir)

            for i in tqdm(self.cid_list):

                result_file = os.path.join(
                    self.pred_dir, 'results_{}.series_{}.npy'.format(i, sub_count))

                if os.path.exists(result_file):
                    continue
                # if i > 0 and 'statistic' in self.model_opts.dict:
                if i > 0 and   self.model_opts.arch == 'statistic':
                    # assert self.model_opts.statistic
                    result0 = str(os.path.join(
                        self.pred_dir, 'results_{}.series_{}.npy'.format(0, sub_count)))
                    shutil.copy(result0, result_file)
                    continue

               
                self.conduct_iter(
                    i, subPack, result_file, innerSaving=True)



    def conduct_iter(self, i, subPack, result_file = None, innerSaving=True):
        try:
            clogger = self.logger_config(
                self.model_fit_dir, 'train', i, subPack.index)
            clogger.critical('*'*80)
            clogger.critical('Dataset: {}\t Model:{} \t H: {}\t Trail: {}'.format(
                self.data_name, self.model_name, self.model_opts.hyper.H, i))
            
            
            sid_hyper = Opt(self.model_opts.hyper)
            sid_hyper.H = subPack.H
            sid_hyper.model_fit_dir = self.model_fit_dir
            sid_hyper.model_name = self.model_name
            sid_hyper.sid = subPack.index
            sid_hyper.sid_name = subPack.name
            sid_hyper.cid = i
            
            if 'sid_hypers' in self.model_opts.dict:
                sid_exConfig = self.model_opts.sid_hypers[subPack.index]
                sid_hyper.update(sid_exConfig)
            
             # loading the best paramters:
            if self.tune:
                best_hyper = self.load_tuning(subPack)
                sid_hyper.update(best_hyper)
                clogger.info("Updating tuning result complete.")
                clogger.critical('-'*80)

            # self.seed = i
            # torch.manual_seed(self.seed)
            # np.random.seed(self.seed)

            clogger.critical(
                'For {}th-batch-trainingLoading, loading the sub-datasets {}'.format(i, subPack.index))
            clogger.critical('-'*80)

            # Attention! sub.H can be different among multiple series in some cases.
  
            model = self.model_import()
            model = model(sid_hyper, clogger)

            clogger.critical('Loading complete.')
            clogger.critical(f'Model: \n{str(model)}')

            fit_info = model.xfit(subPack.train_data, subPack.valid_data)

            _, tgt, pred = model.task_pred(subPack.test_data)

            _tgt = scaler_inverse(subPack.scaler, tgt)
            _pred =  scaler_inverse(subPack.scaler, pred)

            clogger.critical('-'*50)
            
            self.plot_fitInfo(fit_info, subId=subPack.index,
                                  cvId=i)
            
            if result_file is None:
                result_file = os.path.join(
                    self.pred_dir, 'results_{}.series_{}.npy'.format(i, subPack.index))
            
            if innerSaving:
                np.save(result_file,
                        (_tgt, _pred))
                return fit_info
            else:
                return _tgt, _pred, fit_info
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def tuning(self,):
        try:
            self.tune = True

            # check tune condition, if not satisfying, jump tune.
            if 'innerTuning' in self.model_opts.dict:
                if self.model_opts.innerTuning == True:
                    self.tune = False

            if self.tune:
                for sub_count in self.sid_list:
                    subPack = self.data_opts.seriesPack[sub_count]
                # for sub_count, subPack in enumerate(tqdm(self.data_opts.seriesPack)):
                    assert sub_count == subPack.index

                    self.model_fit_dir = self.model_fit_dir_list[sub_count]

                    tuner_dir = os.path.join(self.model_fit_dir, 'tuner')
                    os_makedirs(tuner_dir)
                    tuner_path = os.path.join(
                        tuner_dir, 'series{}.best.pt'.format(sub_count))

                    self.model_opts.tuner.dir = tuner_dir

                    tLogger = self.logger_config(
                        tuner_dir, 'tuning', 'T', sub_count)
                    if not os.path.exists(tuner_path):

                        pT_hyper = Opt()
                        if 'preTuning_model' in self.model_opts.tuner.dict:
                            pT_path = tuner_path.replace(self.model_name, self.model_opts.tuner.preTuning_model)
                            if os.path.exists(pT_path):
                                pT_hyper.merge(torch.load(pT_path))
                                self.model_opts.hyper.update(pT_hyper)
                                
                                tLogger.critical('-'*80)
                                for (arg, value) in pT_hyper.dict.items():
                                    tLogger.info("PreTuning Results:\t %s - %r", arg, value)
                            else:
                                raise ValueError('Non-found preTuning results: {}.\nPlease check the preTuning_model: {}'.format(pT_path, self.model_opts.tuner.preTuning_model))
                        
                        if len(list(self.model_opts.tuning.dict.keys())) > 0:
                            series_tuner = HyperTuner(
                                self.model_opts, tLogger, subPack)
                            best_hyper = series_tuner.conduct() # best_hyper is an Obj
                        else:
                            best_hyper = Opt()
                        # best_hyper = series_tuner.best_config
                        tLogger.critical('-'*80)
                        tLogger.critical('Tuning complete.')
                        
                        pT_hyper.update(best_hyper, ignore_unk=True)
                        torch.save(pT_hyper, tuner_path)
                        
                    else:
                        pT_hyper = torch.load(tuner_path)

                    # best_hyper is a dict type
                    
                    if isinstance(pT_hyper, Mapping):
                        pT_hyper_info = pT_hyper
                    elif isinstance(pT_hyper, object):
                        pT_hyper_info = vars(pT_hyper)
                    else:
                        raise ValueError('Error data type in pT_hyper from: {}'.format(tuner_path))
                                            
                    for (arg, value) in pT_hyper_info.items():
                        tLogger.info("Tuning Results:\t %s - %r", arg, value)
            return self.tune

        except:
            tLogger.exception(
                '{}\nGot an error on tuning.\n{}'.format('!'*50, '!'*50))

    def load_tuning(self, subPack):
        tuner_dir = os.path.join(self.model_fit_dir, 'tuner')
        tuner_path = os.path.join(
            tuner_dir, 'series{}.best.pt'.format(subPack.index))
        best_hyper = torch.load(tuner_path)
        if not os.path.exists(tuner_path):
            raise ValueError(
                'Invalid tuner path: {}'.format(tuner_path))
        return best_hyper

    def plot_fitInfo(self, fit_info, subId, cvId):
        if 'loss_list' in fit_info.dict and 'vloss_list' in fit_info.dict:
            plot_dir = os.path.join(self.model_fit_dir, 'figures', 'loss_curve')
            os_makedirs(plot_dir)
            plot_xfit(fit_info, 'cv{}.series{}'.format(
                cvId, subId), plot_dir)

    def evaluation(self, metrics=['rmse'], force_update=True, naive = 'last', plot_hE = True, remove_outlier = False):
        try:
            # self.eval_dir = os.path.join(self.exp_dir, 'eval')
            
            self.metrics = metrics
            eLogger = set_logger(os.path.join(self.eval_dir, 'h{}.{}.eval.log'.format(self.data_opts.info.H, self.model_name)), '{}.H{}.{}'.format(
                self.data_name, self.data_opts.info.H, self.model_name.upper()), self.logger_level)

            
            
            show_naive = True if naive in ['avg', 'last'] else False

            naive_infos = []
            eval_list = []
            for sub_count in self.sid_list:
                
                naive_info = Opt()
                naive_info.score = {}
                for m in metrics:
                    naive_info.score[m] = ''
                
                if show_naive:
                    # condcut naive method
                    naive_info.H, naive_info.lag_order = self.data_opts.info.H, self.data_opts.info.lag_order
                    naiveF = Naive(naive_info, eLogger, method= naive)
                    subPack = self.data_opts.seriesPack[sub_count]
                    _, n_tgt, n_pred = naiveF.task_pred(subPack.test_data)
                    naive_eval,_ = self.eval_iter(0, sub_count, result_zip=(scaler_inverse(subPack.scaler, n_tgt), scaler_inverse(subPack.scaler, n_pred)))
                    for i, m in enumerate(metrics):
                        naive_info.score[m] = '{:.4g}'.format(naive_eval[0, i])
                naive_infos.append(naive_info)

                self.model_eval_dir = self.model_eval_dir_list[sub_count]
                
                self.measure_dir = os.path.join(
                    self.model_eval_dir, 'eval_results')
                self.pred_dir = os.path.join(self.fit_dir, 'series{}'.format(sub_count), 'h{}'.format(self.data_opts.info.H), self.model_name, 'pred_results')
                self.plot_dir = os.path.join(self.model_eval_dir, 'figures')
                
                os_makedirs(self.pred_dir)
                os_makedirs(self.measure_dir)
                os_makedirs(self.plot_dir)

                s_elogger = set_logger(os.path.join(self.model_eval_dir, 'h{}.{}.eval.log'.format(self.data_opts.info.H, self.model_name)), '{}.H{}.{}'.format(
                self.data_name, self.data_opts.info.H, self.model_name.upper()), self.logger_level)

                # check the mase:
                # if 'mase' in self.metrics:
                #     naivePred = self.get_naivePred(sub_count)
                #     self.data_opts.seriesPack[sub_count].naiveP = naivePred

                ser_eval = []
                for i in self.cid_list:
                    metric_file = os.path.join(
                        self.measure_dir, 'metrics_{}.series_{}.npz'.format(i, sub_count))
                    
                    if os.path.exists(metric_file) and force_update is False:
                        with np.load(metric_file) as data:
                            eval_results, i_th_results = data['score'], data['h_score']
                        # eval_results, i_th_results = np.load(metric_file, allow_pickle=True)
                    else:
                        eval_results, i_th_results = self.eval_iter(
                            i, sub_count)

                    if self.data_opts.info.H > 1 and plot_hE:
                        plot_hError(i_th_results,self.metrics, cid= i, location=os.path.join(self.plot_dir, 'h_e_curve'))

                    
                    s_elogger.critical('*'*80)
                    s_elogger.critical('Dataset: {}\t Model: {}\t H: {}\tSeries-id: {}\t Trail-id: {}'.format(
                        self.data_name, self.model_name, self.data_opts.info.H, sub_count, i))
                    for _i, eval_name in enumerate(self.metrics):
                        s_elogger.critical(
                            'Testing\t{}:\t{:.4g}'.format(eval_name, eval_results[0, _i]))
                    ser_eval.append(eval_results)
                    np.savez(metric_file, score = eval_results, h_score = i_th_results)
                                        
            
                eval_list.append(ser_eval)
                s_elogger.critical('-'*80)

            self.eval_info = Opt()
            self.eval_info.series = {}
            # if self.rep_times > 1:
            for sub_count, sid in enumerate(self.sid_list):
                ser_eval = eval_list[sub_count]
                eLogger.critical('='*80)
                eLogger.critical('Dataset: {}\t Model: {}\t H: {}\t Series-id: {} \t Trail-Nums: {}'.format(
                    self.data_name, self.model_name, self.data_opts.info.H, sid, self.rep_times))

                series_eval_dict = self.evalSeries2dict(ser_eval,remove_outlier )
                # self.eval_info.series.append(series_eval_dict)
                self.eval_info.series[sid] = series_eval_dict
                for metric_name in self.metrics:
                    eLogger.critical('Testing {}\t{}Mean:\t{:.4g}\tMin:\t{:.4g}\tMax:\t{:.4g}\tStd:\t{:.4g}'.format(
                        metric_name.upper(), 
                        '{}Naive:\t{}\t'.format(naive.capitalize() ,naive_infos[sub_count].score[metric_name]) if show_naive else '' ,
                        series_eval_dict[metric_name]['mean'], 
                        series_eval_dict[metric_name]['min'], 
                        series_eval_dict[metric_name]['max'], 
                        series_eval_dict[metric_name]['std']))

            eLogger.critical('@'*80)
            eLogger.critical('Dataset: {}\t Model: {}\t H: {}\t Series-Nums: {}\t Trail-Nums: {}'.format(
                self.data_name, self.model_name, self.data_opts.info.H, len(self.sid_list), len(self.cid_list)))

            # all_eval_list = [item for series in eval_list for item in series]
            all_eval_avg = self.evalData2dict(eval_list,remove_outlier)
            for metric_name in self.metrics:
                eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
                    metric_name, all_eval_avg[metric_name]['mean'], all_eval_avg[metric_name]['std']))

            self.eval_info.all = all_eval_avg

            return self.eval_info
        except:
            eLogger.exception(
                '{}\nGot an error on evaluation.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def evalSeries2dict(self, _eval_list, remove_outlier = False):
        eval_data = np.concatenate(_eval_list, axis=0)

        all_eval_avg = {}
        for i, metric_name in enumerate(self.metrics):
            i_data = eval_data[:, i].tolist()
            
            if len(i_data) > 3 and remove_outlier:
                # try to remove min and max 
                out_tag = IQR_check(eval_data[:, i])
                _i_data = []
                for cid, cid_data in enumerate(i_data):
                    if cid not in out_tag:
                        _i_data.append(cid_data)
                        
                mean = statistics.mean(_i_data)
                std = statistics.stdev(_i_data, mean)
                
            elif len(i_data) > 1:
                mean = statistics.mean(i_data)
                std = statistics.stdev(i_data, mean)
            # if self.rep_times * self.data_opts.info.num_series > 1:
            else:
                mean = i_data[0]
                std = 0

            # eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
            #     metric_name, mean, std))

            all_eval_avg[metric_name] = {}
            all_eval_avg[metric_name]['mean'] = mean
            all_eval_avg[metric_name]['std'] = std
            all_eval_avg[metric_name]['raw'] = i_data
            all_eval_avg[metric_name]['min'] = min(i_data)
            all_eval_avg[metric_name]['max'] = max(i_data)

        return all_eval_avg
    
    def evalData2dict(self, series_list, remove_outlier = False):
        
        all_eval_avg = {}
        
        for i, metric_name in enumerate(self.metrics):
            all_data = []
            for perS in series_list:
                perS_data = np.concatenate(perS, axis=0)
                i_data = perS_data[:, i].tolist()
                if len(i_data) > 3 and remove_outlier:
                    # try to remove min and max 
                    out_tag = IQR_check(perS_data[:, i])
                    _i_data = []
                    for cid, cid_data in enumerate(i_data):
                        if cid not in out_tag:
                            _i_data.append(cid_data)
                else:
                    _i_data = i_data
                
                all_data.extend(_i_data)
            
            if len(all_data) > 1:
                mean = statistics.mean(all_data)
                std = statistics.stdev(all_data, mean)
            else:
                mean = all_data[0]
                std = 0
            
            all_eval_avg[metric_name] = {}
            all_eval_avg[metric_name]['mean'] = mean
            all_eval_avg[metric_name]['std'] = std
            
        return all_eval_avg  

            # eLogger.critical('Testing {}\tMean:\t{:.4g}\tStd:\t{:.4g}'.format(
            #     metric_name, mean, std))
            
    def eval_iter(self, i, sub_count, result_zip=None):

        if result_zip is not None:
            _test_target, _pred = result_zip[0], result_zip[1]
        else:
            # to remove in the next version.
            result_file = os.path.join(
                os.path.join(self.fit_dir, 'series{}'.format(sub_count), 'h{}'.format(self.data_opts.info.H), self.model_name, 'eval_results'), 'results_{}.series_{}.npy'.format(i, sub_count))
            dst_file = os.path.join(self.pred_dir, 'results_{}.series_{}.npy'.format(i, sub_count))
            
            if os.path.exists(result_file):
                shutil.copyfile(result_file, dst_file)
                if os.path.exists(dst_file):
                    os.remove(result_file)
                    result_file = dst_file
            else:
                result_file = dst_file
                            
            _test_target, _pred = np.load(result_file, allow_pickle=True)
            

        eval_results_len = len(self.metrics)
        eval_results = np.zeros((1, eval_results_len))
        i_th_results = np.zeros((self.data_opts.info.H, eval_results_len))
        for i, eval_name in enumerate(self.metrics):
            measure_path = importlib.import_module('task.metric')
            eval = getattr(measure_path, eval_name)
            # if self.data_name == 'mg':
            #     # eval_result = eval(_test_target, _pred, ith = 84)
            #     eval_result = eval(_test_target, _pred)
            # else:
            if eval_name == 'mase':
                # naivePred = self.get_naivePred(sub_count)
                eval_result = eval(_test_target, _pred,
                                   self.data_opts.seriesPack[sub_count].avgNaiveError)
                
                for h in range(self.data_opts.info.H,):
                    i_th_results[h, i] = eval(_test_target, _pred,
                                   self.data_opts.seriesPack[sub_count].avgNaiveError, ith = h + 1 )
            else:
                eval_result = eval(_test_target, _pred)
                for h in range(self.data_opts.info.H,):
                    i_th_results[h, i] = eval(_test_target, _pred, ith = h + 1)
                            
            eval_results[0, i] = eval_result
                    
        return eval_results, i_th_results