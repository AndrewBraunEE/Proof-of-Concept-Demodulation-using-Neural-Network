import argparse
from PyQt5.QtWidgets import QApplication

from commander.app.main import MainWindow

argparser = argparse.ArgumentParser('Launch the EE132A Project')
argparser.add_argument('-v', '--verbose', action='store_true',
                           help='Increase output and log verbosity')

args = argparser.parse_args()
app = QApplication(['EE132A Project'])
mw = MainWindow(args)
mw.show()
mw.jupyter.setFocus()

try:
    app.exec_()
except KeyboardInterrupt:
    pass
