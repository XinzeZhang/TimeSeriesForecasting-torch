from task.parser import get_parser
from task.TaskWrapper import *
from models.training.CNN import GS_CNN

class gscnnTask(Task):
    def __init__(self, args):
        super().__init__(args)
    
    def model_config(self, logger):
        if self.model_arch == 'cnn':
            self.opts.kernel_size = self.opts.steps//4

        if self.opts.model == 'cnn':
            model = GS_CNN(self.opts, logger)
        
        return model

    def conduct(self):
        self.opts.log_level = 3

        times = self.opts.rep_times

        for i in trange(times):

            self.seed = i
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            if os.path.exists(os.path.join(self.opts.task_dir, 'measures.npy')):
                measures = np.load(os.path.join(
                    self.opts.task_dir, 'measures.npy'))
                if measures[i, 0].item() != 0.0:
                    continue

            self.opts.cv = i
            logger = logging.getLogger('{}.cv{}'.format(self.opts.model, i))
            set_logger(os.path.join(self.opts.task_dir,
                                    'train.cv{}.log'.format(i)), logger)
            logger.info(
                'Loading the datasets and model for {}th-batch-training'.format(i))

            # use GPU if available
            if self.tune:
                assert self.best_config is not None
                logger.info(
                'Loading the best config for model')
                self.opts.update(self.best_config)

            model = self.model_config(logger)

            logger.info('Loading complete.')
            logger.info(f'Model: \n{str(model)}')
            # if not os.path.exists(os.path.join(self.opts.task_dir, 'train.cv{}.pth.tar'.format(i+1))) and self.opts.restore:
            model.gs_search(self.train_loader, self.valid_loader)


if __name__ == "__main__":
    
    args = get_parser()
    args.test = True

    # args.cuda = True
    args.datafolder = 'work.esm'
    args.dataset = 'btc'
    args.model = 'cnn'
    args.rep_times = 1

    task = gscnnTask(args)


    task.conduct()