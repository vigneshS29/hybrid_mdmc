from customargparse import HMDMC_ArgumentParser
import sys

def main(argv):

    parser = HMDMC_ArgumentParser()
    parser.HMDMC_parse_args()
    parser.adjust_default_args()
    args = parser.args
    for k,v in args.__dict__.items():
        print(k,v,type(v))
    return

if __name__ == '__main__':
    main(sys.argv[1:])