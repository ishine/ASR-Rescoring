import sys
sys.path.append("../../src")

from util.arg_parser import ArgParser

if __name__ == "__main__":
    arg_parser = ArgParser()
    config = arg_parser.parse()