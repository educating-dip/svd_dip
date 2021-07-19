import os
import socket
import datetime
import torch
import numpy as np
import tensorboardX
from torch.optim import Adam
from torch.nn import MSELoss
from warnings import warn
from functools import partial
from tqdm import tqdm

from .network import UNet
from .utils import poisson_loss, tv_loss, PSNR, normalize, extract_learnable_params


class LRPolicy():
    """
    Learning rate policy implementing linear warmup phase(s), for use with
    the `LambdaLR` scheduler.
    """
    def __init__(self, init_lr, lr, num_warmup_iter, num_iterations):
        self.init_lr = init_lr
        self.lr = lr
        self.num_warmup_iter = num_warmup_iter
        self.num_iterations = num_iterations
        self.lambda_fct = np.ones(self.num_iterations + 1)
        self.restart(0, init_lr, lr, preserve_initial_warmup=False)

    def restart(self, epoch, init_lr, lr, preserve_initial_warmup=True):
        """
        Add a linear warmup phase starting at `epoch`, linearly increasing
        from `init_lr` to `lr`. After the warmup, `lr` is used. Optionally, the
        initial warmup phase is respected by taking the pointwise minimum of
        both rates.
        Note: `epoch` specifies the iteration.
        """
        n = min(self.num_warmup_iter, max(0, self.num_iterations + 1 - epoch))

        self.lambda_fct[epoch:epoch+n] = np.linspace(
                init_lr/self.lr, lr/self.lr, self.num_warmup_iter)[:n]

        self.lambda_fct[epoch+n:] = lr/self.lr

        if preserve_initial_warmup:
            self.lambda_fct[epoch:self.num_warmup_iter] = np.minimum(
                    self.lambda_fct[epoch:self.num_warmup_iter],
                    np.linspace(self.init_lr/self.lr, 1., self.num_warmup_iter)[epoch:])

    def __call__(self, epoch):
        """Note: `epoch` specifies the iteration."""
        return self.lambda_fct[epoch]


class DeepImagePriorReconstructor():
    """
    CT reconstructor applying DIP with TV regularization (see [2]_).
    The DIP was introduced in [1].
    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018, "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           https://doi.org/10.1109/CVPR.2018.00984
    .. [2] D. Otero Baguer, J. Leuschner, M. Schmidt, 2020, "Computed
           Tomography Reconstruction Using Deep Image Prior and Learned
           Reconstruction Methods". Inverse Problems.
           https://doi.org/10.1088/1361-6420/aba415
    """

    def __init__(self, ray_trafo_module, reco_space, observation_space, cfg):

        self.reco_space = reco_space
        self.observation_space = observation_space
        self.cfg = cfg
        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.ray_trafo_module = ray_trafo_module.to(self.device)
        self.init_model()

    def init_model(self):

        input_depth = 1 if not self.cfg.add_init_reco else 2
        output_depth = 1
        scaling_kwargs = {
            'mean_in': self.cfg.stats.mean_fbp,
            'mean_out': self.cfg.stats.mean_gt,
            'std_in': self.cfg.stats.std_fbp,
            'std_out': self.cfg.stats.std_gt
            } if self.cfg.normalize_by_stats else {}

        self.model = UNet(
            input_depth,
            output_depth,
            channels=self.cfg.arch.channels[:self.cfg.arch.scales],
            skip_channels=self.cfg.arch.skip_channels[:self.cfg.arch.scales],
            use_sigmoid=self.cfg.arch.use_sigmoid,
            use_norm=self.cfg.arch.use_norm,
            use_scale_layer = self.cfg.normalize_by_stats,
            scaling_kwargs = scaling_kwargs
            ).to(self.device)

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'DIP+TV'
        logdir = os.path.join(
            self.cfg.log_path,
            current_time + '_' + socket.gethostname() + comment)
        self.writer = tensorboardX.SummaryWriter(logdir=logdir)

    def apply_model_on_test_data(self, net_input):
        test_scaling = self.cfg.get('implicit_scaling_except_for_test_data')
        if test_scaling is not None and test_scaling != 1.:
            if self.cfg.add_init_reco:
                net_input = torch.cat(
                        (test_scaling * net_input[:, 0].unsqueeze(dim=1),
                         net_input[:, 1].unsqueeze(dim=1)), dim=1)
            output = self.model(net_input)
            output = output / test_scaling
        else:
            output = self.model(net_input)

        return output

    def reconstruct(self, noisy_observation, fbp=None, ground_truth=None):

        if self.cfg.torch_manual_seed:
            torch.random.manual_seed(self.cfg.torch_manual_seed)

        self.init_model()
        if self.cfg.load_pretrain_model:
            path = \
                self.cfg.learned_params_path if self.cfg.learned_params_path.endswith('.pt') \
                    else self.cfg.learned_params_path + '.pt'
            path = os.path.join(os.getcwd().partition('src')[0], path)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.model.to(self.device)

        self.model.train()

        if self.cfg.recon_from_randn:
            self.net_input = 0.1 * \
                torch.randn(1, *self.reco_space.shape)[None].to(self.device)
            if self.cfg.add_init_reco:
                self.net_input = \
                    torch.cat([fbp.to(self.device), self.net_input], dim=1)
        else:
            self.net_input = fbp.to(self.device)

        self.init_optimizer()
        self.init_scheduler()
        y_delta = noisy_observation.to(self.device)

        if self.cfg.optim.loss_function == 'mse':
            criterion = MSELoss()
        elif self.cfg.optim.loss_function == 'poisson':
            criterion = partial(poisson_loss,
                                photons_per_pixel=self.cfg.optim.photons_per_pixel,
                                mu_water=self.cfg.optim.mu_water)
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        best_loss = np.inf
        best_output = self.apply_model_on_test_data(self.net_input).detach()

        loss_history = []
        loss_avg_history = []
        last_lr_adaptation_iter = 0

        with tqdm(range(self.cfg.optim.iterations), desc='DIP', disable= not self.cfg.show_pbar) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.apply_model_on_test_data(self.net_input)
                loss = criterion(self.ray_trafo_module(output), y_delta) + self.cfg.optim.gamma * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                self.writer.add_scalar('lr_encoder', self.optimizer.param_groups[0]['lr'], i)
                self.writer.add_scalar('lr_decoder', self.optimizer.param_groups[1]['lr'], i)

                self.scheduler.step()
                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                if self.cfg.optim.use_adaptive_lr:
                    loss_history.append(loss.item())
                    loss_avg_history.append(np.inf if i+1 < self.cfg.optim.adaptive_lr.num_avg_iter else
                                            np.mean(loss_history[-self.cfg.optim.adaptive_lr.num_avg_iter:]))

                    if (i >= last_lr_adaptation_iter + self.cfg.optim.adaptive_lr.num_avg_iter and
                            loss_avg_history[-1] > loss_avg_history[-self.cfg.optim.adaptive_lr.num_avg_iter-1] * (1. - self.cfg.optim.adaptive_lr.min_rel_loss_decrease)):
                        self._adapt_lr(i)
                        last_lr_adaptation_iter = i

                if ground_truth is not None:
                    best_output_psnr = PSNR(best_output.detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(output.detach().cpu(), ground_truth.cpu())
                    pbar.set_postfix({'output_psnr': output_psnr})
                    self.writer.add_scalar('best_output_psnr', best_output_psnr, i)
                    self.writer.add_scalar('output_psnr', output_psnr, i)

                self.writer.add_scalar('loss', loss.item(),  i)
                if i % 1000 == 0:
                    self.writer.add_image('reco', normalize(best_output[0, ...]).cpu().numpy(), i)

        self.writer.close()

        return best_output[0, 0, ...].cpu().numpy()

    def init_optimizer(self):
        """
        Initialize the optimizer.
        """

        self._optimizer = \
            torch.optim.Adam([{'params': extract_learnable_params(self.model,
                             ['down', 'inc']),
                             'lr': self.cfg.optim.encoder.lr},
                             {'params': extract_learnable_params(self.model,
                             ['up', 'scale', 'outc']),
                             'lr': self.cfg.optim.decoder.lr}])

    @property
    def optimizer(self):
        """
        :class:`torch.optim.Optimizer` :
        The optimizer, usually set by :meth:`init_optimizer`, which gets called
        in :meth:`train`.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def init_scheduler(self):

        # always use `self.scheduler` to enable lr changes on checkpoint
        # returns, but set init_lr=lr if `not self.cfg.optim.use_scheduler` to
        # disable warmups
        if self.cfg.optim.use_scheduler:
            init_lr_encoder = self.cfg.optim.encoder.init_lr
            init_lr_decoder = self.cfg.optim.decoder.init_lr
        else:
            init_lr_encoder = self.cfg.optim.encoder.lr
            init_lr_decoder = self.cfg.optim.decoder.lr

        self._lr_policy_encoder = LRPolicy(
                init_lr=init_lr_encoder,
                lr=self.cfg.optim.encoder.lr,
                num_warmup_iter=self.cfg.optim.encoder.num_warmup_iter,
                num_iterations=self.cfg.optim.iterations)
        self._lr_policy_decoder = LRPolicy(
                init_lr=init_lr_decoder,
                lr=self.cfg.optim.decoder.lr,
                num_warmup_iter=self.cfg.optim.decoder.num_warmup_iter,
                num_iterations=self.cfg.optim.iterations)

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                lr_lambda=[self._lr_policy_encoder, self._lr_policy_decoder])

        self.init_lr_fct_encoder = 1.
        self.init_lr_fct_decoder = 1.
        self.lr_fct_encoder = 1.
        self.lr_fct_decoder = 1.

    def _adapt_lr(self, iteration):

        self.init_lr_fct_encoder *= self.cfg.optim.adaptive_lr.get('multiply_init_lr_by', 1.)
        self.init_lr_fct_decoder *= self.cfg.optim.adaptive_lr.get('multiply_init_lr_by', 1.)
        self.lr_fct_encoder *= self.cfg.optim.adaptive_lr.get('multiply_lr_by', 1.)
        self.lr_fct_decoder *= self.cfg.optim.adaptive_lr.get('multiply_lr_by', 1.)

        lr_encoder = self.cfg.optim.encoder.lr * self.lr_fct_encoder
        lr_decoder = self.cfg.optim.decoder.lr * self.lr_fct_decoder
        if (self.cfg.optim.use_scheduler and
                self.cfg.optim.adaptive_lr.restart_scheduler):
            init_lr_encoder = self.cfg.optim.encoder.init_lr * self.init_lr_fct_encoder
            init_lr_decoder = self.cfg.optim.decoder.init_lr * self.init_lr_fct_decoder
        else:
            init_lr_encoder = lr_encoder
            init_lr_decoder = lr_decoder

        self._lr_policy_encoder.restart(iteration, init_lr_encoder, lr_encoder,
                preserve_initial_warmup=not self.cfg.optim.adaptive_lr.restart_scheduler)
        self._lr_policy_decoder.restart(iteration, init_lr_decoder, lr_decoder,
                preserve_initial_warmup=not self.cfg.optim.adaptive_lr.restart_scheduler)

    @property
    def scheduler(self):
        """
        torch learning rate scheduler :
        The scheduler, usually set by :meth:`init_scheduler`, which gets called
        in :meth:`train`.
        """
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value
