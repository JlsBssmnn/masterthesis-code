# from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit

import time
from types import GenericAlias
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QValidator

ConfigType = list[dict[str, any]]

class TupleValidator(QValidator):
    def __init__(self, element_type):
        super().__init__()
        self.element_type = element_type
    
    def validate(self, input, pos):
        if input == '':
            return (QValidator.State.Acceptable, input, pos)
        last_index = len(input) - 1 if input.endswith(',') else len(input)
        elements = input[:last_index].split(',')
        try:
            for element in elements:
                self.element_type(element)
        except:
            return (QValidator.State.Invalid, input, pos)

        return (QValidator.State.Acceptable, input, pos)

class LabelInput(QHBoxLayout):
    def __init__(self, attr_name, attr_type=str, default=''):
        super().__init__()

        self.attr_name = attr_name
        self.attr_type = attr_type

        self.addWidget(QLabel(attr_name))
        self.line_edit = QLineEdit()

        if attr_type == int:
            validator = QIntValidator()
            self.line_edit.setValidator(validator)
        elif attr_type == float:
            validator = QDoubleValidator()
            self.line_edit.setValidator(validator)
        elif type(attr_type) == GenericAlias and attr_type.__origin__ == tuple:
            assert len(attr_type.__args__) == 1
            arg = attr_type.__args__[0]
            validator = TupleValidator(arg)
            self.line_edit.setValidator(validator)
        elif attr_type == str:
            pass
        else:
            raise NotImplementedError(f'The attribute type {self.attr_type} is not implemented')

        self.line_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.line_edit.setText(str(default))
        self.addWidget(self.line_edit)


    def get_state(self):
        try:
            if self.attr_type == str:
                return self.attr_name, self.line_edit.text()
            elif self.attr_type == int:
                return self.attr_name, int(self.line_edit.text())
            elif self.attr_type == float:
                return self.attr_name, float(self.line_edit.text())
            elif self.attr_type == tuple[int]:
                return self.attr_name, tuple([int(x.strip()) for x in self.line_edit.text().split(',')])
            elif self.attr_type == tuple[float]:
                return self.attr_name, tuple([float(x.strip()) for x in self.line_edit.text().split(',')])
            else:
                raise
        except:
            return self.attr_name, None

class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, task):
        super().__init__()
        self.task = task

    def run(self):
        self.task()
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self, config: ConfigType, run_function):
        super().__init__()

        self.setWindowTitle("Config Tuner")
        layout = QVBoxLayout()

        self.attrs = []
        self.run_function = run_function

        for attr_specification in config:
            attr = LabelInput(**attr_specification)
            self.attrs.append(attr)
            layout.addLayout(attr)

        self.run_button = QPushButton("Run function")
        self.run_button.clicked.connect(self.run_button_handler)
        layout.addWidget(self.run_button)

        self.run_time_info = QLabel('')
        layout.addWidget(self.run_time_info)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def get_config(self):
        state = dict()
        for attr in self.attrs:
            name, value = attr.get_state()
            state[name] = value
        return state

    def run_button_handler(self):
        start_time = time.time()
        self.run_time_info.setText('')

        def thread_done():
            self.run_time_info.setText('Ran in %.1fs' % (time.time() - start_time))
            self.run_button.setEnabled(True)

        self.worker = Worker(lambda: self.run_function(self.get_config()))
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        self.run_button.setEnabled(False)
        self.thread.finished.connect(thread_done)


def library_func(config, run_function):
    app = QApplication([])
    window = MainWindow(config, run_function)
    window.show()
    app.exec()
    return window.get_config()

def run_func(config):
    time.sleep(2)
    print(config)

if __name__ == '__main__':
    res = library_func([
            {'attr_name': 'stride', 'attr_type': tuple[int]},
            {'attr_name': 'p_norm', 'attr_type': tuple[float]},
            {'attr_name': 'min_distance', 'attr_type': int, 'default': 1},
            {'attr_name': 'test', 'attr_type': str, 'default': 'hello'},
        ], run_func)

    print('result is', res)
