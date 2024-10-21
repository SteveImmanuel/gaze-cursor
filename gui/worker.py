import traceback

from PyQt5.QtCore import QRunnable, pyqtSignal, QObject


class WorkerSignal(QObject):
    error = pyqtSignal(str)
    success = pyqtSignal(str, str)

class IterateSignal(QObject):
    coordinate = pyqtSignal(int, int)
    complete = pyqtSignal(str)

class KeypointExtractorSignal(QObject):
    error = pyqtSignal(str)



class Worker(QRunnable):

    def __init__(self, function):
        super().__init__()
        self.setAutoDelete(True)
        self.function = function
        self.signal = WorkerSignal()

    def run(self):
        try:
            output = self.function()
            self.signal.success.emit(*output)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.signal.error.emit(str(e))
