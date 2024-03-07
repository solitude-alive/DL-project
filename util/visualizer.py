import os
import ntpath
import time
import torch
import torchvision

from . import util, logger


def save_images(images, image_path):
    """
    Save images to the disk.

    Parameters:
        images (Torch.tensor)    -- a tensor of images (range in [-1, 1])
        image_path (str)         -- the string is used to create image paths

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    assert isinstance(images, torch.Tensor), "images should be a torch tensor"
    assert len(images.size()) == 4, "images should be a 4D tensor"
    images = (images + 1) / 2  # from [-1, 1] to [0, 1]
    nrow = images.shape[0]  # number of images in a row
    torchvision.utils.save_image(images, image_path, nrow=nrow)


class Visualizer:
    """
    This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display,
    and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saving HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.save_img = opt.isTrain  # if the experiment is training, save images.
        self.name = opt.name
        self.saved = False
        self.current_epoch = 0

        # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
        if self.save_img:
            self.img_dir = os.path.join(opt.checkpoints_dir, self.name, 'images')
            logger.log('create image directory %s...' % self.img_dir)
            util.mkdir(self.img_dir)
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """ Reset the self.saved status """
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result=True):
        """
        Save current results to disk.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to disk
        """
        if self.save_img and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.4d_%s.png' % (epoch, label))
                save_images(image, img_path)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f sec, data: %.3f sec) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        logger.log(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        # log to csv
        logger.logkv("epoch", epoch)
        logger.logkv("iters", iters)
        logger.logkvs(losses)
        logger.dumpkvs()
