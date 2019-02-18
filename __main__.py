import argparse
argparser = argparse.ArgumentParser('Launch the EE132A Project')
argparser.add_argument('-v', '--verbose', action='store_true',
                           help='Increase output and log verbosity')
args = argparser.parse_args()

try:
    pass
except KeyboardInterrupt:
    pass
