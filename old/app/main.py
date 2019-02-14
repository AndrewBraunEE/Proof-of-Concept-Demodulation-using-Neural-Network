#This app should be run rooted in trunk with the command:
#python -m commander

import os
import sys
import queue
import logging
import argparse
import threading

from datetime import datetime

from PyQt5.QtCore import QThread, QSettings, QTimer, QItemSelectionModel, pyqtRemoveInputHook
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QShortcut, QHeaderView
from PyQt5.QtGui import QKeySequence

from common import shared

#State Machine imports
from state_machine.state_machine import StateMachine
from state_machine.test import Test

sys.path.append('./commander/app')

import color
from pyconsole import QIPythonWidget
from settings import Settings
from network import Network
from response import ResponseHandler
from command import CommandHandler, CommandDispatcher, CommandDispatcherInterface, CommandButtonInterface
from command_loader import CommandLoader, CommandStager
from load_exporter import LoadInterface
from script_manager import ScriptManager, ScriptsCollection
from interface_tab import InterfaceSelector, AuthInterface
from netlog_reader import NetLogReader
from linetransforms import linetransformer

from ui.Ui_MainWindow import Ui_MainWindow

#imports from libelfin
from libelfin import gse_globals, parsing, utils, siphash, scripting, cpt_builder, cmd_wrappers
from libelfin.commands import fc, pwr, acb, idpu, wd, radio, pc, cpt
from libelfin.new_commands import fc as fc_, pwr as pwr_, acb as acb_, pc as pc_, wd as wd_, radio as radio_, cpt as cpt_

class MainWindow(QMainWindow):

    def __init__(self, args):
        QMainWindow.__init__(self)
        pyqtRemoveInputHook()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.ui_tabs.setCurrentIndex(0)

        self.args = args
        self.windows = {'telemetry': None}
       
        # NOTE Access to self.state in other threads MUST use
        #      the lock self.locks['state']
        
        # TODO Split up 'state' dictionary and 'statistics' dictionary?
        # TODO More fields for interfaces dictionary, e.g. per-interface stats
        # TODO Statistic helper functions, e.g. rolling average for latencies
        self.state = {
                        'gse': {
                            'start_time': datetime.now(),
                            'sent': {'frames': 0, 'bytes': 0},
                            'received': {'frames': 0, 'bytes': 0},
                            'in_sequence': False,
                            'active_sequences': [],
                            'parsing_failures': [] # used only in response.py, TODO: use?
                        }, 
                        'interfaces': {
                            'last_query_time': None,
                            'available': []
                        }
                    }

        self.locks = {'state': threading.Lock()}

        self.queues = {
                    'commands': queue.PriorityQueue(),
                    'new_commands': queue.PriorityQueue(),
                    'received': queue.Queue(),
                    'telemetry': queue.Queue(),
                    'parser': queue.Queue()}

        self.events = {'ready': threading.Event(),
                       'got_send_status': threading.Event(), # TODO: unused, remove?
                       'got_response': threading.Event()}

        self.auth = siphash.ELFINAuth(siphash.DEFAULTKEY, enabled=True)
        
        self._init_logging()
        self._init_settings()
        
        self.network = Network(self)
        self.response_handler = ResponseHandler(self)
        self.command_handler = CommandHandler(self) # TODO: migrate away from CommandHandler
        self.command_dispatcher_ui = CommandDispatcherInterface(self)
        self.command_dispatcher = CommandDispatcher(self, self.command_dispatcher_ui)
        self.command_btns = CommandButtonInterface(self)
        self.loader = CommandLoader(self)
        self.loader_ui = LoadInterface(self)
        self.stager = CommandStager(self)
        self.netlog_reader = NetLogReader(self, self.ui.ui_log_gsed_list)
        
        self.scripts = ScriptsCollection()
        self.script_manager = ScriptManager(self.scripts)

        self.interface_selector = InterfaceSelector(self, self.ui.ui_interfaces_devices_tbl)
        self.auth_interface = AuthInterface(self)
        
        self._init_jupyter()
        self._init_connections()
        self._init_state_machine()

        self.network.start()
        self.response_handler.start()
        self.command_handler.start()
        self.command_dispatcher.start()
        self.netlog_reader.start()
        # state_machine tester
        #self.tester = Test(self.state_machine)
        #self.tester.start()

        # TODO Add network connectivity check
        # TODO Add gsed status request to determine active interfaces
        self.events['ready'].set()

        #list of timers for parsers
        self.timers = queue.Queue()
        
        #On startup, print a nl to console for formatting and request interface status from gsed.
        self.jupyter.append_stream('\n')

        self.command_handler.submit(0, shared.GsedCommandMessage(shared.GsedCmd.DEVICE_SCAN, description = 'gsed scan devices'))
        self.auth_interface.scan()
        self.command_handler.submit(0, shared.GsedCommandMessage(shared.GsedCmd.LOADER_GET_STATUS, description = "get loader status"))
    
    def _init_logging(self):
        """"""

        log_format = '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        self._loglevel = 'DEBUG' if self.args.verbose else 'INFO'
        logging.basicConfig(level=self._loglevel, format=log_format, datefmt=date_format)

        # TODO Logging handlers
        self.log = logging.getLogger('gse')

    def _init_settings(self):
        """Restore previously saved settings."""

        self.settings = Settings(self)
        self.settings.load()

        qt_settings = QSettings('ELFIN', 'GSE')
        try:
            self.restoreGeometry(qt_settings.value('geometry', ''))
            self.restoreGeometry(qt_settings.value('windowState', ''))
        except TypeError:
            self.log.debug('Could not restore geometry or windowState; might be first run')

    def _init_jupyter(self):
        # Stop IPython's flood of debug messages
        logging.getLogger('ipykernel').setLevel(logging.WARNING)
        logging.getLogger('traitlets').setLevel(logging.WARNING)

        self.jupyter = QIPythonWidget(parent=self)
        self.jupyter.pushVariables({'gse': self,
                                    'ui': self.ui,
                                    'self': self.jupyter,
                                    'fc': fc,
                                    'fc_': fc_,
                                    'pwr': pwr,
                                    'pwr_': pwr_,
                                    'acb': acb,
                                    'acb_': acb_,
                                    'idpu': idpu,
                                    'wd': wd,
                                    'wd_': wd_,
                                    'radio': radio,
                                    'radio_': radio_,
                                    'pc': pc,
                                    'pc_': pc_,
                                    'cpt': cpt,
                                    'cpt_': cpt_,
                                    'scripts': self.scripts,
                                    'auth': self.auth,
                                    'sm': self.script_manager,
                                    'loader': self.loader,
                                    'stager': self.stager,
                                    'btn': self.command_btns,
                                    'leet':self.jupyter.leetMode})
        
        self.ui.ui_action_clear_console.triggered.connect(self.jupyter.clearResetTerminal)
        self.ui.ui_console_layout.addWidget(self.jupyter)

        shell = self.jupyter.kernel_manager.kernel.shell
        shell.input_transformer_manager.python_line_transforms.append(linetransformer())        
        
        logging.getLogger('gse.jupyter').setLevel(logging.WARNING)

    def _init_connections(self):
        
        #### Command Loader #####

        #available commands window
        self.ui.ui_commandload_available_tbl.setModel(self.loader)
        self.ui.ui_commandload_available_tbl.verticalHeader().show()
        self.ui.ui_commandload_available_tbl.horizontalHeader().resizeSection(1, 400)
        self.ui.ui_commandload_available_tbl.horizontalHeader().resizeSection(2, 200)
        
        self.ui.ui_commandload_btn_load.pressed.connect(
            self.loader.load_cmd_from_file
        )
        
        #staged commands window
        self.ui.ui_commandload_staged_tbl.setModel(self.stager)
        self.ui.ui_commandload_staged_tbl.verticalHeader().show()
        self.ui.ui_commandload_staged_tbl.horizontalHeader().resizeSection(1, 400)
        self.ui.ui_commandload_staged_tbl.horizontalHeader().resizeSection(2, 200)
        
        self.ui.ui_commandload_btn_stage.pressed.connect(
            lambda : self.move_to_staged() )
        self.ui.ui_commandload_btn_unstage.pressed.connect(
            lambda : self.move_to_loader() )
        self.ui.ui_commandload_btn_delete.pressed.connect(
            lambda: self.loader.remove_selected( self.ui.ui_commandload_available_tbl.selectedIndexes() ))
        self.ui.ui_commandload_btn_export.pressed.connect( self.stager.export )
        self.ui.ui_commandload_btn_up.pressed.connect(
            lambda : self.stager_move_up( self.ui.ui_commandload_staged_tbl.selectedIndexes() ))
        self.ui.ui_commandload_btn_down.pressed.connect(
            lambda : self.stager_move_down( self.ui.ui_commandload_staged_tbl.selectedIndexes() ))
        self.ui.ui_commandload_staged_tbl.doubleClicked.connect(self.stager.edit_timestamp)

        #### Clear GSED Log ####
        self.ui.ui_log_gsed_clear.clicked.connect(self.netlog_reader.clear)
        
        #### Interfaces Tab buttons and table ####

        self.ui.ui_interfaces_devices_tbl.setModel(self.interface_selector)
        self.ui.ui_interfaces_devices_btn_scan.pressed.connect(
            lambda : self.command_handler.submit(0, shared.GsedCommandMessage(shared.GsedCmd.DEVICE_SCAN, description = 'gsed scan devices'))
        )
        self.ui.ui_interfaces_devices_btn_activate.pressed.connect(self.interface_selector.activate_selected)
        self.ui.ui_interfaces_devices_btn_connect_recv.pressed.connect(self.interface_selector.bind_recver_selected)
        self.ui.ui_interfaces_devices_btn_connect_send.pressed.connect(self.interface_selector.bind_sender_selected)
        self.ui.ui_interfaces_sending_kiss_btn_add.pressed.connect(
            lambda : self.interface_selector.bind_recver_kiss(self.ui.ui_interfaces_sending_kiss_edit_host.text(), self.ui.ui_interfaces_sending_kiss_edit_port.text()))
        self.ui.ui_interfaces_auth_btn_load.pressed.connect(self.auth_interface.scan)
        self.ui.ui_interfaces_auth_btn_save.pressed.connect(self.auth_interface.save)

        #### Network Connections ####

        self.network.received_reply_signal.connect(self.handle_remote_reply)
        self.network.received_published_signal.connect(self.handle_remote_publication)
        self.network.error_signal.connect(
            lambda err: self.notify_user(err, color.Colors.MAGENTA)
        )

        #### Quit Application ####

        self.ui.ui_action_quit.triggered.connect(self.close)
        
    def _init_state_machine(self):
        self.state_machine = StateMachine(self.settings['SAT_ID'], self.args)
        self.state_machine.start() # start DB thread

        self.windows['telemetry'] = self.state_machine.window

        self.ui.ui_action_telemetry.toggled.connect(
            lambda checked: self.windows['telemetry'].show() if checked else self.windows['telemetry'].hide()
        )
        self.windows['telemetry'].hide()
        
    def notify_user(self, message, fgcolor=None, duration=1000, everywindow=False):
        self.statusbar_update(message, duration, everywindow)

        if fgcolor:
            message = color.color_text(message, fgcolor)
        message = "\n{}\n".format(message)
        self.jupyter.printText(message, True)

    def statusbar_update(self, message, duration=2000, everywindow=False):

        self.ui.ui_statusbar.showMessage(message, duration)

        if everywindow:
            for win in self.windows.keys():
                self.windows[win].ui.ui_statusbar.showMessage(message, duration)
 
    def handle_remote_reply(self, reply):
        
        if reply.msg_type == shared.ReqRepType.RT_CMD_ACK:
            self.log.debug('Received ack to real-time command for: '+reply.description)       
        else:
            self.log.debug('Unrecognized network operation: %r' % reply)

    def handle_remote_publication(self, pub):

        #received satelite publication
        if shared.PubType.SAT_EVENT.value in pub.msg_type.value:
            if 'bcn' in pub.msg_type.value:
                #received becon packet
                # self.state_machine.parse_beacon(pub.payload, pub.event_time)
                self.log.debug('received beacon from %s' % pub.sat_id)
                return
            
            #if this is an expected response, set internal state to ready to accept more cmds
            if not self.events['ready'].is_set():
                self.events['ready'].set()
            
            self.log.info('Publication received from ELFIN ' + pub.sat_id + ': ' + str(pub.payload))
            
            #check CRC is settings says to
            if self.settings.current['REJECT_INCORRECT_CRC']['value']:
                valid = self.validate_crc(pub.payload)
                if not valid:
                    return False
            
            try:
                timer = self.timers.get(block=False)
                timer.stop()
            except queue.Empty:
                timer = None

            try:
                parser_fn = self.queues['parser'].get(block=False)
            except queue.Empty:
                self.log.debug('No parser found, using default')
                parser_fn = None
                
            try:
                pub_msg, parser_fn = parsing.parse_select(pub.payload, parser_fn)
                
                #print to console if the publication is from our spacecraft
                #This should eventually be changed to be controlled by a filter setting
                if pub.sat_id == self.settings['SAT_ID']:
                    self.jupyter.printText(str(pub_msg), True)
                    self.jupyter.append_stream('\n')
                
            except Exception as e:
                self.log.info('parsing failure occured: '+str(e))
                #debug output
                self.notify_user("Error parsing publication: {}\nPacket:\n{}".format(e), pub.payload)
                
            try:
                pass # self.state_machine.parse_default(pub.payload, pub.event_time)
            except Exception as e:
                self.log.exception(e)

            #if this response was due to a sequence, notify the command dispatcher
            with self.locks['state']:
                seq_flag = self.state['gse']['in_sequence']

            if seq_flag:
                self.command_handler.send_seq_after_delay()

            
        elif pub.msg_type == shared.PubType.GSED_HEARTBEAT:
            #receive heartbeat
            self.log.debug('Gsed heartbeat received')
        elif pub.msg_type == shared.PubType.GSED_DEVICE_SCANNED:
            #update scanned interfaces
            self.log.debug('received gsed interface scan')
            self.interface_selector.new_scan(pub.found)
        elif pub.msg_type == shared.PubType.GSED_INTERFACE_DISPATCH_STATUS or pub.msg_type == shared.PubType.GSED_INTERFACE_DISPATCH_CHANGED:
            self.log.debug('received gsed interface status')
            self.interface_selector.new_status(pub.devices, pub.managers, pub.senders, pub.receivers)
        elif pub.msg_type == shared.PubType.GSED_AUTH_STATUS:
            self.log.debug("Received gsed auth status: {}".format(pub.data))
            self.auth_interface.update(pub.data)
        elif pub.msg_type.value.startswith('gsed/dispatch/'):
            self.command_dispatcher_ui.model.notify(pub)
        elif pub.msg_type.value.startswith('gsed/loader/'):
            self.loader_ui.notify(pub)
        else:
            self.log.debug('Unrecognized network operation: %r' % pub)

    def send_to_dispatch(self, msg, desc = '', send_raw = False):
        """
        takes a msg to send to gsed. msg could be either a real time command or gsed command
        """
        with self.locks['state']:
            #check that no current command is running
            if not self.events['ready'].is_set() or self.state['gse']['in_sequence']:
                self.log.debug('Blocked command dispatch due to running command or sequence')
                self.notify_user('[Sending Unavailable] Waiting for previous command', color.Colors.RED)
                return

        if isinstance(msg, cmd_wrappers.Command):
            self.dispatch_command(msg)

        #send raw bytes
        elif send_raw and isinstance(msg, bytes):
            self.log.debug('sending raw bytes %s'%(msg))
            self.command_handler.submit(0, self.wrap_command(msg, desc, send_raw))
            
        elif isinstance(msg[0], bytes): #rt cmd case

            if msg[1] is not None:
                self.log.debug('Adding parse function {} to list'.format(msg[1]))
                self.queues['parser'].put(msg[1])
            
                #set a timer for popping a parser
                parser_timer = QTimer()
                self.timers.put(parser_timer)
                parser_timer.timeout.connect(self.handle_frame_timeout)
                parser_timer.setSingleShot(True)
                parser_timer.start(self.settings.current['DEFAULT_LAST_FRAME_TIMEOUT']['value'])
            self.command_handler.submit(0, self.wrap_command(msg[0], desc, send_raw))
            
        elif isinstance(msg[0], dict): #sequence of commands
            self.command_handler.submit(0, msg)
        else:
            self.log.info('unrocognized send format')

    def dispatch_command(self, command):
        if not isinstance(command, cmd_wrappers.Command):
            self.log.critical("dispatch_command received a {}, expected Command instance".format(type(command)))
            return
        
        sat_id = self.settings.current['SAT_ID']['value']
        self.command_dispatcher.submit(10, sat_id, command)

    def wrap_command(self, msg, desc = '', send_raw = False):
        """wraps a (bytes, parser) cmd into a RTCommandMessage"""
        if not send_raw:
            command_bytes = self.auth.appendHashToBytes(msg)
            padded_command = utils.pad_command(command_bytes)
        else:
            padded_command = msg

    
        sat_id = self.settings.current['SAT_ID']['value']
        rtc =  shared.RTCommandMessage(padded_command, sat_id, description = desc)

        return rtc        

    #helper funcions that coordinate command loader and stager
    def move_to_staged(self):
        """move selected items from loader to stager"""
        sel = self.ui.ui_commandload_available_tbl.selectedIndexes()
        if self.loader.check_selected(sel):
            removed = self.loader.remove_selected(sel)
            for cmd in reversed(removed):
                self.stager.add_staged(cmd)
            return True
        else:
            return False
        
    def move_to_loader(self):
        """move selected items from loader to stager"""
        sel = self.ui.ui_commandload_staged_tbl.selectedIndexes()
        removed = self.stager.remove_selected(sel)
        for cmd in reversed(removed):
            self.loader.re_add_available(cmd)
        return True

    def stager_move_up(self, selection):
        if not self.stager.move_selected_up(selection):
            return False
        for index in selection:
            #select the newly moved tiles
            newIndex = self.stager.createIndex(index.row()-1, index.column())
            self.ui.ui_commandload_staged_tbl.selectionModel().select(newIndex, QItemSelectionModel.Select)
                
    def stager_move_down(self, selection):
        if not self.stager.move_selected_down(selection):
            return False
        for index in selection:
            #select the newly moved tiles
            newIndex = self.stager.createIndex(index.row()+1, index.column())
            self.ui.ui_commandload_staged_tbl.selectionModel().select(newIndex, QItemSelectionModel.Select)
    
    def handle_frame_timeout(self):
        """
        called when a command which expected a response didn't get one
        """
        if not self.events['ready'].is_set():
            self.events['ready'].set()
        try:
            self.queues['parser'].get(block=False)
            self.timers.get(block=False)
        except queue.empty:
            self.log.debug('queues were empty, this should not happen normally. Check for bugs')
            return
            
        self.notify_user('[Timed out waiting for response]', color.Colors.RED)
        self.jupyter.append_stream('\n')
        self.log.info('Timed out waiting for publication')

    def validate_crc(self, data):
        expected_crc = utils.compute_crc(0, data[:-1])
        received_crc = utils.bytes_to_uint(data[-1:]) 
        
        if expected_crc == received_crc:
            self.log.debug('Valid CRC: got %s == %s' % (received_crc, expected_crc))
            return True
        else:
            invalid_params = (expected_crc, received_crc, utils.bytes_to_hex(data))
            self.log.warning('Invalid CRC (got %s, not %s): %s' % invalid_params)
            return False
        
    def closeEvent(self, event):
      
        quit_msg = "Are you sure you want to exit?"
        reply = QMessageBox.question(self, 'Confirmation', quit_msg,
                                           QMessageBox.Yes,
                                           QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.jupyter.kernel_client.stop_channels()
            self.jupyter.kernel_manager.shutdown_kernel()
            self.deleteLater()

            try:
                self.log.info('Quitting')

                for window in self.windows.keys():
                    self.windows[window].close()

                try:
                    self.settings.save()
                    settings = QSettings('ELFIN', 'GSE')
                    settings.setValue('geometry', self.saveGeometry())
                    settings.setValue('windowState', self.saveState())
                    del settings
                except Exception as e:
                    self.log.warning('%r' % e)

                with self.locks['state']:
                    self.log.debug('Internal state information:\n{}'.format(self.state))

                self.network.stop()
                self.command_handler.stop()
                self.command_dispatcher.stop()
                self.response_handler.stop()
                self.state_machine.stop()
                QMainWindow.closeEvent(self, event)

            except KeyboardInterrupt:
                QMainWindow.closeEvent(self, event)

        else:
            event.ignore()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser('Launch the ELFIN Commander application.')
    argparser.add_argument('-v', '--verbose', action='store_true',
                           help='Increase output and log verbosity')

    args = argparser.parse_args()
    app = QApplication([''])
    
    mw = MainWindow(args)
    mw.show()
    mw.jupyter.setFocus()

    try:
        app.exec_()          
    except KeyboardInterrupt:
        pass
