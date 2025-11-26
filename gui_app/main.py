import os
import sys

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5 import QtWidgets

from gui_app.popup_window import ModelManagerWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ModelManagerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
