import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import _create_model_training_folder


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device,model_path,**params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.model_path = model_path
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["config.yaml", "PreTrain.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader):

        self.model_path = os.path.join("F:/QRF/Test-Model/", self.model_path)

        os.makedirs(self.model_path, exist_ok=True)

        self.initializes_target_network()

        niter = 0
        best_loss = float('inf')
        for epoch_counter in range(self.max_epochs):
            # 初始化 tqdm 进度条
            print("Starting training loop")
            progress_bar = tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch_counter + 1}/{self.max_epochs}"
            )
            total_loss = 0.0
            num_batches = 0

            for batch_idx, ((batch_view_1, batch_view_2), _) in progress_bar:

                batch_view_1 = batch_view_1.to(self.device).float()
                batch_view_1 = batch_view_1.reshape(batch_view_1.shape[0], 1, batch_view_1.shape[1],
                                                    batch_view_1.shape[2])
                batch_view_2 = batch_view_2.to(self.device).float()
                batch_view_2 = batch_view_2.reshape(batch_view_2.shape[0], 1, batch_view_2.shape[1],
                                                    batch_view_2.shape[2])
                loss = self.update(batch_view_1, batch_view_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新目标网络的参数
                self._update_target_network_parameters()
                niter += 1
                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / num_batches
            print("average_loss", average_loss)
            if (epoch_counter + 1) % 20 == 0:
                self.save_model(os.path.join(self.model_path, f'model_epoch_{epoch_counter + 1}.pth'))

        #self.save_model(os.path.join(self.model_path, 'final_model.pth'))
    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
