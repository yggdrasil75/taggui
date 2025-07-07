from PySide6.QtWidgets import (QMessageBox, QDockWidget, QFormLayout, QScrollArea,
							   QAbstractScrollArea, QWidget)
from utils.big_widgets import TallPushButton
from PySide6.QtCore import Slot
from typing import List, Dict
from models.image_list_model import ImageListModel
from models.proxy_image_list_model import ProxyImageListModel
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import FocusedScrollSettingsComboBox, FocusedScrollSettingsDoubleSpinBox
from widgets.image_list import ImageListView, ImageList
from widgets.dup_image_list import ImageGridWidget
from PySide6.QtCore import Qt


class DuplicateFinderWidget(QDockWidget):
	def __init__(self, image_list_model: ImageList):
		super().__init__()
		self.settings = get_settings()
		self.parentlist = image_list_model
		self.image_list_image_width = self.settings.value('image_list_image_width',
            DEFAULT_SETTINGS['image_list_image_width'])
		self.image_list_tag_separator = self.settings.value('tag_separator',
            DEFAULT_SETTINGS['tag_separator'])
		self.dups = ImageListModel(self.image_list_image_width, self.image_list_tag_separator)
		#self.dup_proxy = ProxyImageListModel(self.dups, )
		self.duplist = ImageGridWidget(self)
		self.detectdupButton = TallPushButton('Detect Duplicates')
		self.detectdupButton.clicked.connect(self.find_duplicates)
		self.dupmethod = FocusedScrollSettingsComboBox(key='dedupmodel')
		self.dupmethod.addItems(['md5', 'sha256', 'average', 'phash', 'dhash', 'whash'])
		container = QWidget()
		self.hbox = QFormLayout(container)
		self.hbox.addWidget(self.dupmethod)
		self.hash_size = FocusedScrollSettingsDoubleSpinBox(key='hash_size', default=8, minimum=8, maximum=64)
		self.hash_size.setSingleStep(8)
		self.hash_thresh = FocusedScrollSettingsDoubleSpinBox(key='threshold', default=0, minimum=0, maximum=1)
		self.hbox.addRow(self.detectdupButton)
		self.hbox.addRow('Comparison Method', self.dupmethod)
		self.hbox.addRow('Hash Size', self.hash_size)
		self.hbox.addRow('Threshold', self.hash_thresh)
		self.hbox.addRow(self.duplist)
		self.setObjectName('Duplicate_Finder')
		self.setWindowTitle('Duplicate Finder')
		self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
		scroll_area = QScrollArea()
		scroll_area.setWidgetResizable(True)
		scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
		scroll_area.setWidget(container)
		self.setWidget(scroll_area)

		self.setWindowTitle("Duplicate Image Finder")

	@Slot()
	def find_duplicates(self):
		self.duplist.images = []
		image = self.parentlist.list_view.get_selected_images()[0]
		method = self.dupmethod.currentText()
		duplicates = find_duplicate_images(images=self.parentlist.proxy_image_list_model.sourceModel().images,
						  method=method, hash_size=self.hash_size.value(), threshold=self.hash_thresh.value())
		for duplicate, dupimage in duplicates:
			if image.hash[method] == duplicate:
				self.duplist.add_image(dupimage)

def find_duplicate_images(images: List[Image], method: str = 'md5', hash_size: int = 8, threshold: int = 0) -> Dict[str, Image]:
	"""Finds duplicate images. Now uses Image class methods for hashing."""
	duplicates: Dict[str, List[str]] = {}
	# Select the appropriate hash calculation method
	for image in images:
		if method in ('average', 'phash', 'dhash', 'whash'):
			hash = image.calculate_image_hash(hash_method=method, hash_size=hash_size)
		elif method == 'md5':
			hash = image.calculate_md5_hash()
		elif method == 'sha256':
			hash = image.calculate_sha256_hash()
		else:
			raise ValueError(f"Invalid method: {method}")
		duplicates[hash, image]

	return duplicates