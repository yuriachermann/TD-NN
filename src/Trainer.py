from tqdm import tqdm
import numpy as np
import torch
import time
from . import data_utils
from sklearn.metrics import confusion_matrix


class Trainer():

    def __init__(self,  
                 model, optimizer, criterion, train_loader,
                 valid_loader, epochs, label_counts,
                 print_intermediate_vals=False, gradient_accumulation=8,
                 save_model_on_every_epoch=False) -> None:
        super().__init__()

        # needed for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.gradient_accumulation = gradient_accumulation
        self.label_counts = label_counts

        # for saving and loading
        self.save_folder_path = f"./models/"
        self.save_model_on_every_epoch = save_model_on_every_epoch
        self.print_intermediate_vals = print_intermediate_vals

        self.train_fn = self.train_function

        # losses
        self.train_loss = []
        self.val_loss =  []
        self.loss_iters = []
        self.valid_mIoU = []
        self.valid_mAcc = []
        self.conf_mat = None

    def train_epoch(self, current_epoch):
        """ Training a model for one epoch """

        loss_list = []
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        # grad accumulator running variable
        grad_count = 0
        for i, (images, labels) in progress_bar:
            # Clear gradients w.r.t. parameters

            preds, loss, seg_mask = self.train_fn(images, labels)

            # Calculate Loss: softmax --> cross entropy loss
            loss_list.append(loss.item())

            # Getting gradients w.r.t. parameters
            loss = loss / self.gradient_accumulation
            loss.backward()

            grad_count+= images.shape[0]

            # Updating parameters
            if grad_count >= self.gradient_accumulation:
                self.optimizer.step()
                self.optimizer.zero_grad()
                grad_count = 0

            progress_bar.set_description(f"Epoch {current_epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")

            if i == len(self.train_loader)-1:
                mean_loss = np.mean(loss_list)
                progress_bar.set_description(f"Epoch {current_epoch+1} Iter {i+1}: mean loss {mean_loss.item():.5f}. ")

        return mean_loss, loss_list


    @torch.no_grad()
    def eval_model(self):
        """ Evaluating the model for either validation or test """
        loss_list = []
        Accs = []
        epsilon = 1e-6

        if self.label_counts != None:
            self.conf_mat = torch.zeros(self.label_counts, self.label_counts)
        else:
            self.conf_mat == None

        # discard gradient for inference
        with torch.no_grad():
            for images, labels in self.valid_loader:

                outputs, loss, seg_mask = self.train_fn(images, labels)

                loss_list.append(loss.item())

                # mIoU
                seg_mask = seg_mask.view(-1)
                preds = outputs.view(-1)

                # prediction to one hot encoding
                one_hot = torch.sigmoid(preds)
                one_hot[one_hot >= 0.5] = 1
                one_hot[one_hot < 0.5] = 0

                if self.label_counts!= None:
                    self.conf_mat += confusion_matrix(
                        y_true=seg_mask.cpu().numpy(), y_pred=one_hot.cpu().numpy(),
                        labels=np.arange(0, self.label_counts, 1)
                    )

                # compute mAcc
                num_correct = torch.sum(one_hot == seg_mask)
                total_predictions = seg_mask.shape[0]
                Accs.append((num_correct/total_predictions).to("cpu"))

            iou = self.conf_mat.diag() / (self.conf_mat.sum(axis=1) + self.conf_mat.sum(axis=0) - self.conf_mat.diag() + epsilon)
            mIoU = iou.mean()

            mAcc = sum(Accs) / len(Accs)
            loss = np.mean(loss_list)
        return mIoU, mAcc, loss


    def train_model(self):
        """ Training a model for a given number of epochs"""

        start = time.time()
        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):

            # validation epoch
            self.model.eval()  # important for dropout and batch norms
            mIoU, mAcc, loss = self.eval_model()
            self.valid_mIoU.append(mIoU)
            self.valid_mAcc.append(mAcc)
            self.val_loss.append(loss)

            # # training epoch
            self.model.train()  # important for dropout and batch norms
            mean_loss, cur_loss_iters = self.train_epoch(epoch)
            self.train_loss.append(mean_loss)


            self.loss_iters = self.loss_iters + cur_loss_iters

            if self.print_intermediate_vals: # and epoch % 5 == 0 or epoch==self.epochs-1):
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"    Train loss: {round(mean_loss, 5)}")
                print(f"    Valid loss: {round(loss, 5)}")
                print(f"    mIoU: {mIoU}%")
                print(f"    mAcc: {mAcc}%")
                print("\n")

            if self.save_model_on_every_epoch == True:
                self.save_model(epoch)
            
        if self.save_model_on_every_epoch == False:
                self.save_model(epoch)

        end = time.time()
        print(f"Training completed after {(end-start)/60:.2f}min")

    def train_function(self, images, labels):

        images = images.to(self.device)
        labels = labels.to(self.device).to(torch.long)

        output = self.model(images)

        loss = self.criterion(output.squeeze(), labels.squeeze().long())

        return output, loss, labels

    def save_model(self, current_epoch):
        # save model
        data_utils.save_model(
            self.model,
            self.optimizer,
            current_epoch,
            [self.train_loss, self.val_loss, self.loss_iters, self.valid_mIoU, self.valid_mAcc, self.conf_mat],
            savepath=(self.save_folder_path + f"epoch_{current_epoch}.pth"),
            )


    def load_model(self, path):
        self.model, self.optimizer, self.start_epoch, self.stats = data_utils.load_model(
            self.model,
            self.optimizer,
            path,
            self.device
        )
        self.train_loss, self.val_loss, self.loss_iters, self.valid_mIoU, self.valid_mAcc, self.conf_mat = self.stats

    def count_model_params(self):
        """ Counting the number of learnable parameters in a nn.Module """
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return num_params
