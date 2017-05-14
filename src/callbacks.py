from keras.callbacks import Callback
from datetime import datetime
import pandas as pd


class MonitorGrowth(Callback):
    '''Callback that builds a dataframe storing the train and validation
     loss and accuracy, and the length of train time, associated with each
     batch and epoch during training'''

    def __init__(self, gene_name, max_seconds):
        super(Callback, self).__init__()
        self.gene_name = gene_name
        self.time_stopping_limit = max_seconds

    def on_train_begin(self, logs={}):
        # get the phenotype dataframe started
        self.epoch = 0
        self.batch = 0
        self.train_start_time = datetime.now()
        # set first start_time with datetime

    def on_epoch_begin(self, epoch, logs={}):
        #self.epoch_start_time = datetime.now()
        self.epoch_start_time = datetime.now()

    def on_batch_begin(self, batch, logs={}):
        #self.batch_start_time = datetime.now()
        self.batch_start_time = datetime.now()

    def on_batch_end(self, batch, logs={}):
        batch_time = datetime.now() - self.batch_start_time

        #batch_time_diff = current_time-start_time
        new_entry = {'gene_name': [self.gene_name], 'epoch': [self.epoch], 'batch': [self.batch],
                     'epoch_train_loss': [None], 'batch_train_loss': [logs.get('loss')],
                     'epoch_train_acc': [None], 'batch_train_acc': [logs.get('acc')],
                     'epoch_val_loss': [None], 'epoch_val_acc': [None],
                     'batch_time_sec': [batch_time.total_seconds()],
                     'epoch_time_sec': [None]}
        temp_df = pd.DataFrame.from_dict(new_entry)

        if self.batch == 0:
            self.batch_df = temp_df
        else:
            self.batch_df = pd.concat([self.batch_df, temp_df], axis=0)

        self.batch = self.batch + 1

    def on_epoch_end(self, epoch, logs={}):
        # check to see if training should stop based on time limit
        #(originally had this in on_batch_end, but found it completes the epoch anyway)
        train_time = datetime.now() - self.train_start_time
        if train_time.total_seconds() > self.time_stopping_limit:
            self.model.stop_training = True
            print('_______Stopping after %s seconds.' %
                  self.time_stopping_limit)

        # update dataframe with training results
        epoch_time = datetime.now() - self.epoch_start_time
        self.batch_df['epoch_train_loss'] = [
            logs.get('loss')] * (self.batch)
        self.batch_df['epoch_train_acc'] = [
            logs.get('acc')] * (self.batch)
        self.batch_df['epoch_val_loss'] = [
            logs.get('val_loss')] * (self.batch)
        self.batch_df['epoch_val_acc'] = [
            logs.get('val_acc')] * (self.batch)
        self.batch_df['epoch_time_sec'] = [
            epoch_time.total_seconds()] * (self.batch)

        if self.epoch == 0:
            self.df = self.batch_df
        else:
            self.df = pd.concat([self.df, self.batch_df], axis=0)

        self.batch = 0
        self.epoch = self.epoch + 1
