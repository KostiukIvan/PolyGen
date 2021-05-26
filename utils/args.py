import sys

def get_args():
    args, params = sys.argv[1::2], sys.argv[2::2]
    args = [arg.replace('-', '') for arg in args]
    return dict(zip(args, params))