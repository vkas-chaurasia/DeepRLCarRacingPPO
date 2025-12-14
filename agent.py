import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from network import Net

class PPOAgent():
    """
    Agent for training
    """

    def __init__(self, gamma=0.99, seed=0, img_stack=4, clip_param=0.1, lambda_=0.95, c1=1, c2=0.01, ppo_epoch=10, lr=0.001, buffer_capacity=2000, batch_size=128, device='cpu', epochs=10, folder_name=""):
        self.gamma = gamma
        self.seed = seed
        self.clip_param = clip_param
        self.lambda_ = lambda_
        self.c1 = c1
        self.c2 = c2
        self.ppo_epoch = ppo_epoch
        self.lr = lr
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.img_stack = img_stack
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        
        self.writer = SummaryWriter(log_dir=f"runs/CarRacing_PPO_{epochs}")
        self.training_step = 0
        self.transition = np.dtype([('s', np.float64, (self.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (self.img_stack, 96, 96))])
        self.folder_name = folder_name
        self._build_model()

    def _build_model(self):
        self.net = Net(self.img_stack).float().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.max_moving_average_score = -np.inf


    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        
        alpha, beta = alpha.cpu(), beta.cpu()
        dist = Beta(alpha, beta)              
        action = dist.sample().numpy().squeeze()
        return action, dist.log_prob(torch.tensor(action)).sum().item()

    def save_param(self, epoch=None):
        if not self.folder_name:
             return
        torch.save(self.net.state_dict(), f'{self.folder_name}/{self.folder_name}_ppo_net_params.pkl')
        if epoch:
            torch.save(self.net.state_dict(), f'{self.folder_name}/{self.folder_name}_ppo_net_params.pkl')

    def save_max_moving_average_param(self, moving_average_score):
        if not self.folder_name:
             return
        if self.max_moving_average_score < moving_average_score:
            self.max_moving_average_score = moving_average_score
            torch.save(self.net.state_dict(), f'{self.folder_name}/{self.folder_name}_ppo_net_params_max_mov_average.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.float).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.float).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.float).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.float).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.float).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.net(s_)[1]
            # Advantage Estimation
            adv = target_v - self.net(s)[1]

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])  #log-exp  new_policy/old_policy

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]  #clipped objective
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])

                # Entropy loss (encourages exploration)
                entropy_loss = dist.entropy().mean()

                loss = action_loss + self.c1 * value_loss - self.c2 * entropy_loss

                # Log losses in TensorBoard
                self.writer.add_scalar("Loss/Policy_Loss", action_loss.item(), self.training_step)
                self.writer.add_scalar("Loss/Value_Loss", value_loss.item(), self.training_step)
                self.writer.add_scalar("Loss/Entropy_Loss", entropy_loss.item(), self.training_step)
                self.writer.add_scalar("Policy/Mean_Advantage", adv.mean().item(), self.training_step)
                self.writer.add_scalar("Policy/Mean_Ratio", ratio.mean().item(), self.training_step)
                self.writer.add_scalar("Policy/Clip_Fraction", (ratio > (1 + self.clip_param)).float().mean().item(), self.training_step)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class TestPPOAgent():
    def __init__(self, final_epoch_testing=False, param_file_path="", max_average_file_path="", device=None):
        self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.net = Net(img_stack=4).float().to(self.device)
        self.load_param(final_epoch_testing, param_file_path, max_average_file_path)
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha.cpu() / (alpha.cpu() + beta.cpu())
        action = action.squeeze().cpu().numpy()
        return action
    
    def load_param(self, final_epoch_testing, param_file_path, max_average_file_path):
        if max_average_file_path:
            self.net.load_state_dict(torch.load(max_average_file_path, weights_only=True, map_location=self.device))
        if final_epoch_testing and param_file_path:
            self.net.load_state_dict(torch.load(param_file_path, weights_only=True, map_location=self.device))
