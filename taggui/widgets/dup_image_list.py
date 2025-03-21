from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QPixmap, QAction, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QListView, QMenu, QWidget,
                               QVBoxLayout, QMessageBox, QStyledItemDelegate)

from utils.image import Image  # Import your Image class


class ThumbnailDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        image: Image = index.data(Qt.UserRole)  # Get the Image object
        ratio = index.data(Qt.UserRole + 1)  # Ratio is stored in a separate role

        if image and image.thumbnail:
            # Convert QImage to QPixmap for drawing
            pixmap = QPixmap.fromImage(image.thumbnail)
            scaled_pixmap = pixmap.scaled(option.rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = option.rect.x() + (option.rect.width() - scaled_pixmap.width()) / 2
            y = option.rect.y() + (option.rect.height() - scaled_pixmap.height() - 20) / 2
            painter.drawPixmap(int(x), int(y), scaled_pixmap)


        ratio_text = f"{ratio:.2f}%"
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        text_rect = option.rect.adjusted(0, option.rect.height() - 20, 0, 0)
        painter.drawText(text_rect, Qt.AlignCenter, ratio_text)

    def sizeHint(self, option, index):
        return QSize(150, 170)


class ImageGridWidget(QWidget):
    tags_merged = Signal(list)  # Signal for merged tags
    image_added = Signal(Image)
    image_removed = Signal(Image)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.images = []  # Store Image objects directly
        self.ratios = {}  # Store ratios using Image objects as keys

        self.list_view = QListView()
        self.list_view.setViewMode(QListView.IconMode)
        self.list_view.setResizeMode(QListView.Adjust)
        self.list_view.setSpacing(10)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setMovement(QListView.Static)
        self.thumbnail_delegate = ThumbnailDelegate()
        self.list_view.setItemDelegate(self.thumbnail_delegate)

        self.model = QStandardItemModel(self)
        self.list_view.setModel(self.model)

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_view)
        self.setLayout(layout)

        self.list_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_view.customContextMenuRequested.connect(self.show_context_menu)


    def add_image(self, image: Image, ratio: float = 0.0):
        """Adds an Image object to the grid."""
        if image in self.images:  # Prevent duplicates based on Image object
            return

        if not image.thumbnail:
            if not image.load_thumbnail():
                print(f"Error: Could not load thumbnail for {image.path}")
                return
                
        self.images.append(image)
        self.ratios[image] = ratio

        item = QStandardItem()
        item.setData(image, Qt.UserRole)  # Store the Image object
        item.setData(ratio, Qt.UserRole + 1)  # Store ratio in a separate role
        item.setToolTip(str(image.path))  # Show full path on hover. convert to string
        item.setFlags(item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        self.model.appendRow(item)
        self.image_added.emit(image)

    def remove_image(self, image: Image):
        """Removes an image from the grid by its Image object."""
        if image not in self.images:
            return

        index = self.images.index(image)
        self.images.remove(image)
        del self.ratios[image]  # Remove the ratio
        self.model.removeRow(index)
        self.image_removed.emit(image)


    def clear_images(self):
        """Clears all images from the grid."""
        self.images.clear()
        self.ratios.clear()
        self.model.clear()

    def show_context_menu(self, position):
        indexes = self.list_view.selectedIndexes()
        if not indexes:
            return

        menu = QMenu(self)
        merge_action = QAction("Merge", self)
        merge_action.triggered.connect(self.merge_selected_image)
        menu.addAction(merge_action)

        menu.exec(self.list_view.viewport().mapToGlobal(position))


    def _get_image_from_index(self, index):
        """Helper to get Image object from model index."""
        return index.data(Qt.UserRole)


    @Slot()
    def merge_selected_image(self):
        indexes = self.list_view.selectedIndexes()
        if not indexes: return

        image: Image = self._get_image_from_index(indexes[0])
        if not image: return

        tags = image.tags

        try:
            Path(image.path).unlink()  # Delete the image file
            caption_path = image.path.with_suffix(".txt")
            if caption_path.exists():
                caption_path.unlink()
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
            return

        self.remove_image(image)
        return tags

    def get_images(self) -> list:
        """
        returns a list of the `Image`s currently added to the widget
        """
        return self.images