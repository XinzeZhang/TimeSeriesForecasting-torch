'''
Defines the neural network, loss function and metrics for single class
Todo: adding choice for single class and multiple class
'''

from data_process.util import init_metrics, get_metrics, update_metrics, final_metrics
from tqdm import trange
from data_process.util import deep_ar_loss_fn as loss_fn
from data_process.util import plot_all_epoch, plot_eight_windows,plot_xfit, os_makedirs
from data_process.util import savebest_checkpoint, load_checkpoint, save_dict_to_json
# import logging
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


# from data_process.util import set_.logger


# logger = logging.getLogger('DeepAR.Net')


class DeepAR(nn.Module):
    def __init__(self, params, logger):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super(DeepAR, self).__init__()
        self.params = params
        self.logger = logger

        for (arg, value) in params.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

        self.lstm = nn.LSTM(input_size=1+params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_presigma = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, 1)
        self.distribution_sigma = nn.Softplus()
        self.optimizer = optim.Adam(self.parameters(), lr=params.learning_rate)
        self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, params.step_lr, gamma=0.9)

        self.num_epochs = self.params.num_epochs

        self.params.plot_dir = os.path.join(params.model_dir, 'figures')
        # create missing directories
        os_makedirs(self.params.plot_dir)

        if self.params.device == torch.device('cpu'):
            self.logger.info('Not using cuda...')
        else:
            self.logger.info('Using Cuda...')
            self.to(self.params.device)

    def forward(self, x, hidden, cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]): LSTM c from time step t
        '''
        # onehot_embed = self.embedding(idx) #TODO: is it possible to do this only once per window instead of per step?
        # lstm_input = torch.cat((x, onehot_embed), dim=2)
        # output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = hidden.permute(
            1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        # softplus to make sure standard deviation is positive
        sigma = self.distribution_sigma(pre_sigma)
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size, self.params.lstm_hidden_dim, device=self.params.device)

    def get_v(self, x_batch):
        # x_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        batch_size = x_batch.shape[0]
        v_input= torch.zeros((batch_size, 2),dtype =torch.float32)
        for i in range(batch_size):
            nonzero_sum = (x_batch[i, 1:self.params.steps, 0]!=0).sum()
            if nonzero_sum.item() == 0:
                v_input[i,0]=0
            else:
                v_input[i,0]=torch.true_divide(x_batch[i, 1:self.params.steps, 0].sum(),nonzero_sum)+1
                x_batch[i,:,0] = x_batch[i,:,0] / v_input[i,0]
            
        return x_batch, v_input

    def fit_epoch(self, train_loader, epoch):
        self.train()
        loss_epoch = np.zeros(len(train_loader))
        # Train_loader:
        # x_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        # idx ([batch_size]): one integer denoting the time series id;
        # labels_batch ([batch_size, train_window]): z_{1:T}.
        for i, (x_batch, label_batch) in enumerate(train_loader):
            self.optimizer.zero_grad()
            batch_size = x_batch.shape[0]
            # batch normalize
            x_batch, v_batch = self.get_v(x_batch)
            label_batch = label_batch/v_batch[:,0].reshape(-1,1)

            x_batch = x_batch.permute(1, 0, 2).to(
                torch.float32).to(self.params.device)  # not scaled
            label_batch = label_batch.permute(1, 0).to(
                torch.float32).to(self.params.device)  # not scaled

            loss = torch.zeros(1, device=self.params.device)
            hidden = self.init_hidden(batch_size)
            cell = self.init_cell(batch_size)

            for t in range(self.params.train_window):
                # if z_t is missing, replace it by output mu from the last time step
                zero_index = (x_batch[t, :, 0] == 0)
                if t > 0 and torch.sum(zero_index) > 0:
                    x_batch[t, zero_index, 0] = mu[zero_index]
                x = x_batch[t].unsqueeze_(0).clone()
                mu, sigma, hidden, cell = self(x, hidden, cell)
                loss += loss_fn(mu, sigma, label_batch[t])

            loss.backward()
            self.optimizer.step()
            loss = loss.item() / self.params.train_window  # loss per timestep
            loss_epoch[i] = loss
            # if i % 1000 == 0:
            #     # todo
            #     test_metrics = self.evaluate(test_loader, epoch, sample=self.params.sampling)
            #     self.train()
            #     self.logger.info(f'train_loss: {loss}')
            # if i == 0:
            #     self.logger.info(f'train_loss: {loss}')

        self.epoch_scheduler.step()
        self.logger.info(f'train_loss: {loss}')
        return loss_epoch

    def xfit(self, train_loader, test_loader, restore_file=None):
        if restore_file is not None and os.path.exists(restore_file) and self.params.restore:
            self.logger.info(
                'Restoring parameters from {}'.format(restore_file))
            load_checkpoint(restore_file, self, self.optimizer)

        best_test_ND = float('inf')
        train_len = len(train_loader)
        ND_summary = np.zeros(self.num_epochs)
        loss_summary = np.zeros((train_len * self.num_epochs))
        loss_avg = np.zeros((self.num_epochs))
        vloss_avg = np.zeros_like(loss_avg)

        epoch = 0
        for epoch in trange(self.num_epochs):
            self.logger.info(
                'Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            # test_len = len(test_loader)
            # print(test_len)
            # loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, loss_fn, train_loader,  test_loader, self.params, epoch)
            loss_epoch = self.fit_epoch(train_loader, epoch)
            loss_summary[epoch * train_len:(epoch + 1) * train_len] = loss_epoch
            # todo
            loss_avg[epoch] = loss_epoch.mean()
            test_metrics, vloss_epoch_i = self.evaluate(
                test_loader, epoch, sample=self.params.sampling)
            vloss_avg[epoch] = vloss_epoch_i

            ND_summary[epoch] = test_metrics['ND']
            is_best = ND_summary[epoch] <= best_test_ND

            # Save weights
            if is_best:
                self.logger.info('- Found new best state')
                best_test_ND = ND_summary[epoch]
                best_json_path = os.path.join(
                    self.params.model_dir, 'metrics_test_best_weights.json')
                save_dict_to_json(test_metrics, best_json_path)

                savebest_checkpoint({
                    'epoch': epoch,
                    'cv': self.params.cv,
                    'state_dict': self.state_dict(),
                    'optim_dict': self.optimizer.state_dict()}, checkpoint=self.params.model_dir)
                self.logger.info(
                    'Checkpoint saved to {}'.format(self.params.model_dir))

            self.logger.info('Current Best ND is: %.5f' % best_test_ND)

            # plot_all_epoch(
            #     ND_summary[:epoch + 1], self.params.dataset + '_ND', self.params.plot_dir)
        
        plot_all_epoch(loss_summary[:(
            epoch + 1) * train_len], self.params.dataset + '_loss', self.params.plot_dir)
        plot_xfit(loss_avg,vloss_avg,self.params.dataset + '_loss', self.params.plot_dir)
            # last_json_path = os.path.join(
            #     self.params.model_dir, 'metrics_test_last_weights.json')
            # save_dict_to_json(test_metrics, last_json_path)


    def point_predict(self, x, using_best=True):
        '''
        x: (numpy.narray) shape: [sample, full-len, dim]
        return: (numpy.narray) shape: [sample, prediction-len]
        Todo: adding choice for single class and multiple class
        '''
        # x_batch: shape: [full-len, sample, dim]
        best_pth = os.path.join(self.params.model_dir, 'best.cv{}.pth.tar'.format(self.params.cv))
        if os.path.exists(best_pth) and using_best:
            self.logger.info(
                'Restoring best parameters from {}'.format(best_pth))
            load_checkpoint(best_pth, self, self.optimizer)
        x = torch.tensor(x)
        x, v_batch = self.get_v(x)
        x_batch = x.permute(1, 0, 2).to(
            torch.float32).to(self.params.device)
        v_batch = v_batch.to(self.params.device)
        batch_size = x_batch.shape[1]
        input_mu = torch.zeros(
            batch_size, self.params.predict_start, device=self.params.device)  # scaled
        input_sigma = torch.zeros(
            batch_size, self.params.predict_start, device=self.params.device)  # scaled
        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)

        prediction = torch.zeros(
            batch_size, self.params.predict_steps, device=self.params.device)

        for t in range(self.params.predict_start):
            # if z_t is missing, replace it by output mu from the last time step
            zero_index = (x_batch[t, :, 0] == 0)
            if t > 0 and torch.sum(zero_index) > 0:
                x_batch[t, zero_index, 0] = mu[zero_index]

            mu, sigma, hidden, cell = self(
                x_batch[t].unsqueeze(0), hidden, cell)
            input_mu[:, t] = mu * v_batch[:, 0] + v_batch[:, 1]
            input_sigma[:, t] = sigma * v_batch[:, 0]

        for t in range(self.params.predict_steps):
            mu_de, sigma_de, hidden, cell = self(
                x_batch[self.params.predict_start + t].unsqueeze(0), hidden, cell)
            prediction[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
            if t < (self.params.predict_steps - 1):
                x_batch[self.params.predict_start + t + 1, :, 0] = mu_de

        return prediction.cpu().detach().numpy()


    def evaluate(self, test_loader, plot_num, sample=True, plot = False):
        self.eval()
        with torch.no_grad():
            # plot_batch = np.random.randint(len(test_loader)-1)
            plot_batch = 0

            summary_metric = {}
            raw_metrics = init_metrics(sample=sample)

            # Test_loader:
            # x_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
            # id_batch ([batch_size]): one integer denoting the time series id;
            # v ([batch_size, 2]): scaling factor for each window;
            # labels ([batch_size, train_window]): z_{1:T}.
            vloss_epoch = np.zeros(len(test_loader))
            for i, (x_batch, labels) in enumerate(test_loader):
                x_batch, v_batch = self.get_v(x_batch)
                label_batch = labels.detach().clone()
                label_batch = label_batch/v_batch[:,0].reshape(-1,1)
                label_batch = label_batch.permute(1, 0).to(
                torch.float32).to(self.params.device)  # not scaled

                x_batch = x_batch.permute(1, 0, 2).to(
                    torch.float32).to(self.params.device)
                # id_batch = id_batch.unsqueeze(0).to(self.params.device)
                v_batch = v_batch.to(torch.float32).to(self.params.device)
                labels = labels.to(torch.float32).to(self.params.device)
                batch_size = x_batch.shape[1]
                input_mu = torch.zeros(
                    batch_size, self.params.predict_start, device=self.params.device)  # scaled
                input_sigma = torch.zeros(
                    batch_size, self.params.predict_start, device=self.params.device)  # scaled
                vloss = torch.zeros(1, device=self.params.device)
                hidden = self.init_hidden(batch_size)
                cell = self.init_cell(batch_size)

                for t in range(self.params.predict_start):
                    # if z_t is missing, replace it by output mu from the last time step
                    zero_index = (x_batch[t, :, 0] == 0)
                    if t > 0 and torch.sum(zero_index) > 0:
                        x_batch[t, zero_index, 0] = mu[zero_index]

                    mu, sigma, hidden, cell = self(
                        x_batch[t].unsqueeze(0).clone(), hidden, cell)

                    vloss += loss_fn(mu,sigma,label_batch[t])
                    input_mu[:,t] = v_batch[:, 0] * mu + v_batch[:, 1]
                    input_sigma[:,t] = v_batch[:, 0] * sigma
                    # input_mu[:, t] = mu
                    # input_sigma[:, t] = sigma

                vloss = vloss.item() / self.params.train_window
                vloss_epoch[i] = vloss



                if sample:
                    samples, sample_mu, sample_sigma = self.test(
                        x_batch, v_batch,hidden, cell, sampling=True)
                    raw_metrics = update_metrics(raw_metrics, input_mu, input_sigma, sample_mu, labels,
                                                 self.params.predict_start, samples, relative=self.params.relative_metrics)
                else:
                    sample_mu, sample_sigma = self.test(
                        x_batch, v_batch,hidden, cell)
                    raw_metrics = update_metrics(raw_metrics, input_mu, input_sigma, sample_mu,
                                                 labels, self.params.predict_start, relative=self.params.relative_metrics)

                if i == plot_batch and plot:
                    if sample:
                        sample_metrics = get_metrics(
                            sample_mu, labels, self.params.predict_start, samples, relative=self.params.relative_metrics)
                    else:
                        sample_metrics = get_metrics(
                            sample_mu, labels, self.params.predict_start, relative=self.params.relative_metrics)
                    # select 10 from samples with highest error and 10 from the rest
                    # hard coded to be 10
                    top_10_nd_sample = (-sample_metrics['ND']
                                        ).argsort()[:batch_size // 10]
                    chosen = set(top_10_nd_sample.tolist())
                    all_samples = set(range(batch_size))
                    not_chosen = np.asarray(list(all_samples - chosen))
                    if batch_size < 100:  # make sure there are enough unique samples to choose top 10 from
                        random_sample_10 = np.random.choice(
                            top_10_nd_sample, size=10, replace=True)
                    else:
                        random_sample_10 = np.random.choice(
                            top_10_nd_sample, size=10, replace=False)
                    if batch_size < 12:  # make sure there are enough unique samples to choose bottom 90 from
                        random_sample_90 = np.random.choice(
                            not_chosen, size=10, replace=True)
                    else:
                        random_sample_90 = np.random.choice(
                            not_chosen, size=10, replace=False)
                    combined_sample = np.concatenate(
                        (random_sample_10, random_sample_90))

                    label_plot = labels[combined_sample].data.cpu().numpy()
                    predict_mu = sample_mu[combined_sample].data.cpu().numpy()
                    predict_sigma = sample_sigma[combined_sample].data.cpu(
                    ).numpy()
                    plot_mu = np.concatenate(
                        (input_mu[combined_sample].data.cpu().numpy(), predict_mu), axis=1)
                    plot_sigma = np.concatenate(
                        (input_sigma[combined_sample].data.cpu().numpy(), predict_sigma), axis=1)
                    plot_metrics = {_k: _v[combined_sample]
                                    for _k, _v in sample_metrics.items()}
                    plot_eight_windows(self.params.plot_dir, plot_mu, plot_sigma, label_plot,
                                       self.params.test_window, self.params.predict_start, plot_num, plot_metrics, sample)

            summary_metric = final_metrics(raw_metrics, sampling=sample)
            metrics_string = '; '.join('{}: {:05.3f}'.format(
                k, v) for k, v in summary_metric.items())
            self.logger.info('- Full test metrics: ' + metrics_string)
        return summary_metric, vloss_epoch.mean()

    def test(self, x, v_batch, hidden, cell, sampling=False):
        batch_size = x.shape[1]
        # x,v_batch = self.get_v(x)

        if sampling:
            samples = torch.zeros(self.params.sample_times, batch_size, self.params.predict_steps,
                                  device=self.params.device)
            for j in range(self.params.sample_times):
                decoder_hidden = hidden
                decoder_cell = cell
                for t in range(self.params.predict_steps):
                    mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                         decoder_hidden, decoder_cell)
                    gaussian = torch.distributions.normal.Normal(
                        mu_de, sigma_de)
                    pred = gaussian.sample()  # not scaled
                    samples[j, :, t] = pred * v_batch[:, 0] + v_batch[:, 1]
                    # samples[j, :, t] = pred
                    if t < (self.params.predict_steps - 1):
                        x[self.params.predict_start + t + 1, :, 0] = pred

            sample_mu = torch.median(samples, dim=0)[0]
            sample_sigma = samples.std(dim=0)
            return samples, sample_mu, sample_sigma

        else:
            decoder_hidden = hidden
            decoder_cell = cell
            sample_mu = torch.zeros(
                batch_size, self.params.predict_steps, device=self.params.device)
            sample_sigma = torch.zeros(
                batch_size, self.params.predict_steps, device=self.params.device)
            for t in range(self.params.predict_steps):
                mu_de, sigma_de, decoder_hidden, decoder_cell = self(x[self.params.predict_start + t].unsqueeze(0),
                                                                     decoder_hidden, decoder_cell)
                sample_mu[:, t] = mu_de * v_batch[:, 0] + v_batch[:, 1]
                sample_sigma[:, t] = sigma_de * v_batch[:, 0]
                # sample_mu[:, t] = mu_de
                # sample_sigma[:, t] = sigma_de
                if t < (self.params.predict_steps - 1):
                    x[self.params.predict_start + t + 1, :, 0] = mu_de
            return sample_mu, sample_sigma
