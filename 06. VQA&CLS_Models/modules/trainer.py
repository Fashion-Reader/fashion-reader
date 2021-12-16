import os
from tqdm import tqdm
import torch


class CustomTrainer():

    """ CustomTrainer
        epoch에 대한 학습 및 검증 절차 정의

    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        metric_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
        logger (`logger`)
    """

    def __init__(self, train_type, model, device, loss_fn, metric_fn, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.train_type = train_type
        self.model = model
        self.device = device
        self.logger = logger
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()

        # History - predict
        self.train_target_pred_list = list()
        self.train_target_list = list()

        self.validation_target_pred_list = list()
        self.validation_target_list = list()

        # Output
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        self.train_score = 0

        self.validation_loss_mean = 0
        self.validation_loss_sum = 0
        self.validation_score = 0

    def train_epoch(self, dataloader, epoch_index=0, verbose=True, logging_interval=1):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
            verbose (boolean)
            logging_interval (int)
        """
        # feature를 cls하는 model train
        if self.train_type == 'feature':
            self.model.train()
            for batch_index, inputs, print_label, neck_label in enumerate(tqdm(dataloader)):
                inputs = inputs.to(self.device)
                print_label = print_label.to(self.device)
                neck_label = neck_label.to(self.device)
                outputs1, outputs2 = self.model(inputs)
                self.optimizer.zero_grad()

                # Loss
                loss1 = self.loss_fn(outputs1, print_label)
                loss2 = self.loss_fn(outputs2, neck_label)
                loss = (loss1+loss2)/2
                batch_loss_mean = loss
                batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
                self.train_batch_loss_mean_list.append(batch_loss_mean.item())
                self.train_loss_sum += batch_loss_sum

                # Metric
                print_pred = outputs1.argmax(dim=-1)
                neck_pred = outputs2.argmax(dim=-1)
                print_acc_item = (print_label == print_pred).sum().item()
                neck_acc_item = (neck_label == neck_pred).sum().item()
                batch_score = (print_acc_item+neck_acc_item)/2
                self.train_batch_score_list.append(batch_score)

                # Update
                loss.backward()
                self.optimizer.step()

                # Log verbose
                if verbose & (batch_index % logging_interval == 0):
                    msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                    self.logger.info(msg) if self.logger else print(msg)
                if batch_index % 50 == 0:
                    msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                    print(msg)
            self.train_loss_mean = self.train_loss_sum / len(dataloader)
            self.train_score = sum(self.train_batch_score_list) / \
                len(self.train_batch_score_list)
            msg = f'Epoch {epoch_index}, Train, Mean loss: {self.train_loss_mean}, Score: {self.train_score}'
            self.logger.info(msg) if self.logger else print(msg)

        # style을 classification하는 model을 train
        elif self.train_type == 'style':
            self.model.train()
            for batch_index, inputs, style_label in enumerate(tqdm(dataloader)):
                inputs = inputs.to(self.device)
                style_label = style_label.to(self.device)
                outputs = self.model(inputs)
                self.optimizer.zero_grad()

                # Loss
                batch_loss_mean = self.loss_fn(outputs, style_label)
                batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
                self.train_batch_loss_mean_list.append(batch_loss_mean.item())
                self.train_loss_sum += batch_loss_sum

                # Metric
                style_pred = outputs.argmax(dim=-1)
                style_acc_item = (style_label == style_pred).sum().item()
                batch_score = style_acc_item
                self.train_batch_score_list.append(batch_score)

                # Update
                batch_loss_mean.backward()
                self.optimizer.step()

                # Log verbose
                if verbose & (batch_index % logging_interval == 0):
                    msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                    self.logger.info(msg) if self.logger else print(msg)
                if batch_index % 50 == 0:
                    msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                    print(msg)

            self.train_loss_mean = self.train_loss_sum / len(dataloader)
            self.train_score = sum(self.train_batch_score_list) / \
                len(self.train_batch_score_list)
            msg = f'Epoch {epoch_index}, Train, Mean loss: {self.train_loss_mean}, Score: {self.train_score}'
            self.logger.info(msg) if self.logger else print(msg)

        # VQA를 train
        elif self.train_type == 'vqa':
            self.model.train()
            for batch_index, data in enumerate(dataloader):
                q_bert_ids = data['ids'].to(self.device)
                q_bert_mask = data['mask'].to(self.device)
                target = data['answer'].to(self.device)
                imgs = data['image'].to(self.device)

                self.optimizer.zero_grad()

                # Loss
                target_pred = self.model(q_bert_ids, q_bert_mask, imgs)
                batch_loss_mean = self.loss_fn(target_pred, target)
                batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
                self.train_batch_loss_mean_list.append(batch_loss_mean.item())
                self.train_loss_sum += batch_loss_sum

                # Metric
                target_list = target.cpu().tolist()
                target_pred = torch.argmax(target_pred, dim=1)
                target_pred_list = target_pred.cpu().tolist()
                batch_score = self.metric_fn(target_list, target_pred_list)
                self.train_batch_score_list.append(batch_score)

                # Update
                batch_loss_mean.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # History - predict
                self.train_target_list.extend(target_list)
                self.train_target_pred_list.extend(target_pred_list)
                # Log verbose
                if verbose & (batch_index % logging_interval == 0):
                    msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                    self.logger.info(msg) if self.logger else print(msg)
                if batch_index % 50 == 0:
                    msg = f"Epoch {epoch_index} train batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                    print(msg)

            self.train_loss_mean = self.train_loss_sum / len(dataloader)
            self.train_score = self.metric_fn(
                self.train_target_list, self.train_target_pred_list)
            msg = f'Epoch {epoch_index}, Train, Mean loss: {self.train_loss_mean}, Score: {self.train_score}'
            self.logger.info(msg) if self.logger else print(msg)

    def validate_epoch(self, dataloader, epoch_index=0, verbose=True, logging_interval=1):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
            verbose (boolean)
            logging_interval (int)
        """
        if self.train_type == 'feature':
            self.model.eval()
            with torch.no_grad():
                for batch_index, inputs, print_label, neck_label in enumerate(tqdm(dataloader)):
                    inputs = inputs.to(self.device)
                    print_label = print_label.to(self.device)
                    neck_label = neck_label.to(self.device)
                    outputs1, outputs2 = self.model(inputs)
                    self.optimizer.zero_grad()

                    # Loss
                    loss1 = self.loss_fn(outputs1, print_label)
                    loss2 = self.loss_fn(outputs2, neck_label)
                    loss = (loss1+loss2)/2
                    batch_loss_mean = loss
                    batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
                    self.validation_batch_loss_mean_list.append(
                        batch_loss_mean.item())
                    self.validation_loss_sum += batch_loss_sum

                    # Metric
                    print_pred = outputs1.argmax(dim=-1)
                    neck_pred = outputs2.argmax(dim=-1)
                    print_acc_item = (print_label == print_pred).sum().item()
                    neck_acc_item = (neck_label == neck_pred).sum().item()
                    batch_score = (print_acc_item+neck_acc_item)/2
                    self.validation_batch_score_list.append(batch_score)
                    # Log verbose
                    if verbose & (batch_index % logging_interval == 0):
                        msg = f"Epoch {epoch_index} validation batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                        self.logger.info(msg) if self.logger else print(msg)
                    if batch_index % 50 == 0:
                        msg = f"Epoch {epoch_index} validation batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                        print(msg)
                self.validation_loss_mean = self.validation_loss_sum / \
                    len(dataloader)
                self.validation_score = sum(self.validation_batch_score_list) / \
                    len(self.validation_batch_score_list)
                msg = f'Epoch {epoch_index}, Validation, Mean loss: {self.validation_loss_mean}, Score: {self.validation_score}'
                self.logger.info(msg) if self.logger else print(msg)

        elif self.train_type == 'style':
            self.model.eval()
            with torch.no_grad():
                for batch_index, inputs, style_label in enumerate(tqdm(dataloader)):
                    inputs = inputs.to(self.device)
                    style_label = style_label.to(self.device)
                    outputs = self.model(inputs)
                    self.optimizer.zero_grad()

                    # Loss
                    batch_loss_mean = self.loss_fn(outputs, style_label)
                    batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
                    self.validation_batch_loss_mean_list.append(
                        batch_loss_mean.item())
                    self.validation_loss_sum += batch_loss_sum

                    # Metric
                    style_pred = outputs.argmax(dim=-1)
                    style_acc_item = (style_label == style_pred).sum().item()
                    batch_score = style_acc_item
                    self.validation_batch_score_list.append(batch_score)

                    # Log verbose
                    if verbose & (batch_index % logging_interval == 0):
                        msg = f"Epoch {epoch_index} validation batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                        self.logger.info(msg) if self.logger else print(msg)
                    if batch_index % 50 == 0:
                        msg = f"Epoch {epoch_index} validation batch {batch_index}/{len(dataloader)}: {batch_index * dataloader.batch_size}/{len(dataloader.dataset)} mean loss: {batch_loss_mean} score: {batch_score}"
                        print(msg)
                self.validation_loss_mean = self.validation_loss_sum / \
                    len(dataloader)
                self.validation_score = sum(self.validation_batch_score_list) / \
                    len(self.validation_batch_score_list)
                msg = f'Epoch {epoch_index}, Validation, Mean loss: {self.validation_loss_mean}, Score: {self.validation_score}'
                self.logger.info(msg) if self.logger else print(msg)

        elif self.train_type == 'vqa':
            self.model.eval()

            with torch.no_grad():
                for batch_index, data in enumerate(tqdm(dataloader)):
                    q_bert_ids = data['ids'].to(self.device)
                    q_bert_mask = data['mask'].to(self.device)
                    target = data['answer'].to(self.device)
                    imgs = data['image'].to(self.device)

                    self.optimizer.zero_grad()

                    # Loss
                    target_pred = self.model(q_bert_ids, q_bert_mask, imgs)
                    batch_loss_mean = self.loss_fn(target_pred, target)
                    batch_loss_sum = batch_loss_mean.item() * dataloader.batch_size
                    self.validation_batch_loss_mean_list.append(
                        batch_loss_mean.item())
                    self.validation_loss_sum += batch_loss_sum

                    # Metric
                    target_list = target.cpu().tolist()
                    target_pred = torch.argmax(target_pred, dim=1)
                    target_pred_list = target_pred.cpu().tolist()
                    batch_score = self.metric_fn(target_list, target_pred_list)
                    self.validation_batch_score_list.append(batch_score)
                    # History - predict
                    self.validation_target_list.extend(target_list)
                    self.validation_target_pred_list.extend(target_pred_list)
                    # Log verbose

                self.validation_loss_mean = self.validation_loss_sum / \
                    (len(dataloader) * dataloader.batch_size)
                self.validation_score = self.metric_fn(
                    self.validation_target_list, self.validation_target_pred_list)
                msg = f'Epoch {epoch_index}, Validation, Mean loss: {self.validation_loss_mean}, Score: {self.validation_score}'
                print(msg)

    def clear_history(self):
        """ 한 epoch 종료 후 history 초기화
            Examples:
                >>for epoch_index in tqdm(range(EPOCH)):
                >>    trainer.train_epoch(dataloader=train_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.validate_epoch(dataloader=validation_dataloader, epoch_index=epoch_index, verbose=False)
                >>    trainer.clear_history()
        """

        # History - loss
        self.train_batch_loss_mean_list = list()
        self.train_batch_score_list = list()

        self.validation_batch_loss_mean_list = list()
        self.validation_batch_score_list = list()

        # History - predict
        self.train_target_pred_list = list()
        self.train_target_list = list()

        self.validation_target_pred_list = list()
        self.validation_target_list = list()

        # Output
        self.train_loss_mean = 0
        self.train_loss_sum = 0
        self.train_score = 0

        self.validation_loss_mean = 0
        self.validation_loss_sum = 0
        self.validation_score = 0
