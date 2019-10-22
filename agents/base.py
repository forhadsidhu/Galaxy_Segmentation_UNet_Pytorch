import logging


class BaseAgent:
    '''
    This base class will contain the base function to be overloaded by any agent you will implement
    '''

    def __init__(self,config):
        self.config=config
        self.logger=logging.getLogger("Agent")

    def load_checkpoint(self,file_name):
        """
        latest checkpoint loader
        :param file_name:
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self,filename="checkpoint.pth.tar",is_best=0):
        """
        checkpoint svaer
        :param filename:
        :param is_best:
        :return:
        """
        raise NotImplementedError
    def run(self):
        """
        the main operator
        :return:
        """
        raise NotImplementedError
    def train(self):
        """
        The main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError
    def finalize(self):
        """
        Finalizes all the operations of 2 main classes of the process the operator and the data loader

        :return:
        """
        raise NotImplementedError
