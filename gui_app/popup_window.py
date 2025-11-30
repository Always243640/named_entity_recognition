from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from gui_app.model_workflows import (
    MODEL_OPTIONS,
    UnknownModelError,
    evaluate_selected_model,
    predict_entities,
    train_selected_model,
)


class WorkerSignals(QtCore.QObject):
    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(str)
    confusion = QtCore.pyqtSignal(object)


class TrainWorker(QtCore.QThread):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.signals = WorkerSignals()

    def run(self):
        try:
            def callback(message: str, progress: int):
                if message:
                    self.signals.log.emit(message)
                self.signals.progress.emit(progress)

            train_selected_model(self.model_name, callback)
            self.signals.finished.emit("模型训练完成！")
        except UnknownModelError as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:  # pragma: no cover - UI safeguard
            self.signals.error.emit(f"训练过程中发生错误: {exc}")


class EvaluateWorker(QtCore.QThread):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.signals = WorkerSignals()

    def run(self):
        try:
            def callback(message: str, progress: int):
                if message:
                    self.signals.log.emit(message)
                self.signals.progress.emit(progress)

            _, confusion_data = evaluate_selected_model(self.model_name, callback)
            self.signals.confusion.emit(confusion_data)
            self.signals.finished.emit("模型评估完成！")
        except UnknownModelError as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:  # pragma: no cover - UI safeguard
            self.signals.error.emit(f"评估过程中发生错误: {exc}")


class InferenceWorker(QtCore.QThread):
    def __init__(self, model_name: str, text: str):
        super().__init__()
        self.model_name = model_name
        self.text = text
        self.signals = WorkerSignals()

    def run(self):
        try:
            def callback(message: str, progress: int):
                if message:
                    self.signals.log.emit(message)
                self.signals.progress.emit(progress)

            result = predict_entities(self.model_name, self.text, callback)
            self.signals.result.emit(result)
            self.signals.finished.emit("实体识别完成！")
        except UnknownModelError as exc:
            self.signals.error.emit(str(exc))
        except ValueError as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:  # pragma: no cover - UI safeguard
            self.signals.error.emit(f"识别过程中发生错误: {exc}")


class ModelManagerWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("命名实体识别模型管理")
        icon_path = Path(__file__).resolve().parent.parent / "imgs" / "biLSTM_NER.png"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        self.resize(760, 820)
        self._init_ui()
        self.train_worker = None
        self.eval_worker = None
        self.predict_worker = None

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("可视化弹窗 - 模型管理")
        title_font = QtGui.QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        form_layout = QtWidgets.QFormLayout()

        self.dataset_field = QtWidgets.QLineEdit("DataNER")
        self.dataset_field.setReadOnly(True)
        form_layout.addRow("数据集：", self.dataset_field)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(MODEL_OPTIONS)
        form_layout.addRow("模型：", self.model_combo)

        layout.addLayout(form_layout)

        button_layout = QtWidgets.QHBoxLayout()
        self.train_button = QtWidgets.QPushButton("重新训练")
        self.train_button.clicked.connect(self.start_training)
        self.eval_button = QtWidgets.QPushButton("模型评测")
        self.eval_button.clicked.connect(self.start_evaluate)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.eval_button)
        layout.addLayout(button_layout)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("训练或评测日志将在此处显示...")
        layout.addWidget(self.log_view)

        confusion_group = QtWidgets.QGroupBox("混淆矩阵")
        confusion_layout = QtWidgets.QVBoxLayout(confusion_group)
        self.confusion_label = QtWidgets.QLabel("混淆矩阵图将在此显示")
        self.confusion_label.setAlignment(QtCore.Qt.AlignCenter)
        self.confusion_label.setMinimumHeight(260)
        self.confusion_label.setStyleSheet(
            "border: 1px solid #dcdcdc; background-color: #fafafa;"
        )
        self.confusion_label.setScaledContents(True)
        confusion_layout.addWidget(self.confusion_label)
        layout.addWidget(confusion_group)

        inference_group = QtWidgets.QGroupBox("实体识别")
        inference_layout = QtWidgets.QVBoxLayout(inference_group)

        self.predict_input = QtWidgets.QTextEdit()
        self.predict_input.setPlaceholderText("请输入需要标注的文本，每行一条输入")
        inference_layout.addWidget(self.predict_input)

        self.predict_output = QtWidgets.QTextEdit()
        self.predict_output.setReadOnly(True)
        self.predict_output.setPlaceholderText("模型识别的实体将在此处显示")
        inference_layout.addWidget(self.predict_output)

        self.predict_button = QtWidgets.QPushButton("实体识别")
        self.predict_button.clicked.connect(self.start_inference)
        inference_layout.addWidget(self.predict_button)

        layout.addWidget(inference_group)

    def _set_buttons_enabled(self, enabled: bool):
        self.train_button.setEnabled(enabled)
        self.eval_button.setEnabled(enabled)
        self.predict_button.setEnabled(enabled)

    def _connect_worker(self, worker, *, finished_handler=None, result_handler=None):
        worker.signals.log.connect(self.append_log)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(self.append_log)
        if finished_handler:
            worker.signals.finished.connect(finished_handler)
        if result_handler and hasattr(worker.signals, "result"):
            worker.signals.result.connect(result_handler)
        if hasattr(worker.signals, "confusion"):
            worker.signals.confusion.connect(self.show_confusion_matrix)
        worker.signals.finished.connect(lambda _: self._set_buttons_enabled(True))
        worker.signals.error.connect(self.handle_error)
        worker.signals.error.connect(lambda _: self._set_buttons_enabled(True))

    def append_log(self, message: str):
        if not message:
            return
        self.log_view.append(message)
        cursor = self.log_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_view.setTextCursor(cursor)

    def handle_error(self, message: str):
        self.append_log(message)
        QtWidgets.QMessageBox.warning(self, "错误", message)

    def start_training(self):
        selected_model = self.model_combo.currentText()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self._set_buttons_enabled(False)

        self.train_worker = TrainWorker(selected_model)
        self._connect_worker(self.train_worker)
        self.train_worker.start()

    def start_evaluate(self):
        selected_model = self.model_combo.currentText()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.confusion_label.setText("正在评估并生成混淆矩阵图...")
        self.confusion_label.setPixmap(QtGui.QPixmap())
        self._set_buttons_enabled(False)

        self.eval_worker = EvaluateWorker(selected_model)
        self._connect_worker(self.eval_worker)
        self.eval_worker.start()

    def start_inference(self):
        selected_model = self.model_combo.currentText()
        text = self.predict_input.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "提示", "请输入需要标注的文本")
            return

        self.log_view.clear()
        self.predict_output.clear()
        self.progress_bar.setValue(0)
        self._set_buttons_enabled(False)

        self.predict_worker = InferenceWorker(selected_model, text)
        self._connect_worker(self.predict_worker, result_handler=self.display_prediction)
        self.predict_worker.start()

    def display_prediction(self, content: str):
        if content:
            self.predict_output.setPlainText(content)

    def _create_confusion_matrix_pixmap(
        self, tags, matrix, cell_size: int = 60, margin: int = 120
    ) -> QtGui.QPixmap:
        if not tags or not matrix:
            return QtGui.QPixmap()

        width = margin + cell_size * len(tags)
        height = margin + cell_size * len(tags)
        image = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        image.fill(QtCore.Qt.white)

        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        label_font = painter.font()
        label_font.setPointSize(10)
        painter.setFont(label_font)
        pen = QtGui.QPen(QtCore.Qt.black)
        painter.setPen(pen)

        for idx, tag in enumerate(tags):
            top_rect = QtCore.QRectF(
                margin + idx * cell_size, margin - cell_size, cell_size, cell_size
            )
            left_rect = QtCore.QRectF(
                margin - cell_size, margin + idx * cell_size, cell_size, cell_size
            )
            painter.drawText(top_rect, QtCore.Qt.AlignCenter, tag)
            painter.drawText(left_rect, QtCore.Qt.AlignCenter, tag)

        max_value = max((max(row) for row in matrix if row), default=0)
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                rect = QtCore.QRectF(
                    margin + col_idx * cell_size,
                    margin + row_idx * cell_size,
                    cell_size,
                    cell_size,
                )
                if max_value:
                    intensity = int(200 * value / max_value)
                    painter.fillRect(rect, QtGui.QColor(70, 130, 180, 55 + intensity))
                painter.drawRect(rect)
                painter.drawText(rect, QtCore.Qt.AlignCenter, str(value))

        painter.end()
        return QtGui.QPixmap.fromImage(image)

    def show_confusion_matrix(self, data):
        if not data or len(data) != 2:
            self.confusion_label.setText("未生成混淆矩阵图")
            self.confusion_label.setPixmap(QtGui.QPixmap())
            return

        tags, matrix = data
        pixmap = self._create_confusion_matrix_pixmap(tags, matrix)
        if pixmap.isNull():
            self.confusion_label.setText("混淆矩阵生成失败")
            return

        self.confusion_label.setPixmap(pixmap)
