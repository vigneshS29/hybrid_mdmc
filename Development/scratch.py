import argparse

class custom_argparse(argparse.ArgumentParser):

    def __init__(self):
        argparse.ArgumentParser.__init__(self)
        self.args.add_default_args()

    def add_default_args(self):
        self.args.add_argument(dest='thelabel')
        self.args.add_argument('-thenumber', dest='thenumber')

    def custom_parse_args(self):
        self.args.args = self.args.parse_args()

    def adjust_default_args(self):
        self.args.args.thelabel = str(self.args.args.thelabel)
        self.args.args.thenumber = int(float(self.args.args.thenumber))



