import time

from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

from util import logger


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.save_epoch_freq = opt.save_epoch_freq
        self.display_freq = opt.display_freq
        self.batch_size = opt.batch_size

        self.dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(self.dataset)  # get the number of images in the dataset.
        logger.log('The number of training images = %d' % dataset_size)

        self.model = create_model(opt)  # create a model given opt.model and other options
        self.model.setup(opt)  # regular setup: load and print networks; create schedulers

        self.visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

        self.iter_count = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        self.iter_count_total = 0  # the total number of training iterations across all epochs
        self.epoch_count = 1    # the starting epoch count
        self.epoch_total = opt.n_epochs + opt.n_epochs_decay    # the total number of training epochs
        self.time_start_iter = None  # the time of each iteration
        self.time_start_epoch = None  # the time of each epoch
        self.time_data = None
        self.time_comp = None

    def train(self):
        for epoch in range(self.epoch_count, self.epoch_total + 1):
            self.iter_count = 0
            self.time_start_epoch = time.time()  # timer for entire epoch
            self.time_start_iter = time.time()  # timer for data loading per iteration

            self.visualizer.reset()  # reset: make sure it saves the results at least once every epoch
            self.model.update_learning_rate()  # update learning rates in the beginning of every epoch

            for i, data in enumerate(self.dataset):  # inner loop within one epoch
                self.time_data = time.time() - self.time_start_iter
                self.forward(data)

            self.visualize(epoch)

    def forward(self, data):
        self.iter_count += self.batch_size
        self.iter_count_total += self.batch_size
        time_comp = time.time()

        self.model.set_input(data)  # unpack data from dataset and apply preprocessing
        self.model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

        self.time_comp = time.time() - time_comp
        self.time_start_iter = time.time()

    def visualize(self, epoch):
        losses = self.model.get_current_losses()
        self.visualizer.print_current_losses(epoch, self.iter_count_total, losses, self.time_comp, self.time_data)

        if epoch % self.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logger.log('saving the model at the end of epoch %d, iters %d' % (epoch, self.iter_count))
            self.model.save_networks(epoch)

        if epoch % self.display_freq == 0:   # display images
            self.visualizer.display_current_results(self.model.get_current_visuals(), epoch)

        logger.log('saving the model at the end of epoch %d' % epoch)
        self.model.save_networks('latest')

        logger.log('End of epoch %d / %d \t Time Taken: %d sec'
                   % (epoch, self.epoch_total, time.time() - self.time_start_epoch))
