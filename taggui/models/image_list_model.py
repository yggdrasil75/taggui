import random
import re
import sys
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Set, Tuple
import time
import os

import exifread
from PySide6.QtCore import (QAbstractListModel, QModelIndex, QSize, Qt, Signal,
                            Slot)
from PySide6.QtGui import QIcon, QImageReader, QPixmap, QImage
from PySide6.QtWidgets import QMessageBox
import pillow_jxl
from PIL import Image as pilimage  # Import Pillow's Image class


from utils.image import Image
from utils.jxlutil import get_jxl_size
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.utils import get_confirmation_dialog_reply, pluralize

UNDO_STACK_SIZE = 32

def pil_to_qimage(pil_image):
    """Convert PIL image to QImage properly"""
    pil_image = pil_image.convert("RGBA")
    data = pil_image.tobytes("raw", "RGBA")
    qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
    return qimage

def get_os_thumbnail(file_path: str, image_width: int) -> QIcon | None:
    """
    Attempts to retrieve a thumbnail from the OS thumbnail cache.

    Args:
        file_path: The path to the image file.
        image_width: The desired width of the thumbnail.

    Returns:
        A QIcon if the OS thumbnail is found and loaded successfully,
        otherwise None.
    """

    # Windows Thumbnail Cache (requires pywin32)
    if os.name == 'nt':
        try:
            import win32com.client
            import pythoncom

            # Ensure COM is initialized for this thread (if not already)
            pythoncom.CoInitialize()
            shell = win32com.client.Dispatch("Shell.Application")
            folder_path, file_name = os.path.split(file_path)
            folder = shell.NameSpace(folder_path)
            if folder: #check the folder exists
                item = folder.ParseName(file_name)
                if item:
                    # Thumbnail size constants (approximate)
                    # 0: Small (96x96)
                    # 1: Medium (256x256)
                    # 2: Large (256x256, same as medium)
                    # 3: Extra Large (platform dependent, often 256 but can be higher)
                    # 4: Jumbo (256x256)
                    thumb = folder.GetDetailsOf(item, 179)  #179 for thumbnail.
                    if thumb: #check we got something.
                        #The thumbnail *is* text representing the file path.
                        #So, we need to load it.
                        #This is the most correct way, creating a pixmap, and QIcon.
                        thumb_pixmap = QPixmap(thumb)
                        if not thumb_pixmap.isNull():
                            thumb_pixmap = thumb_pixmap.scaledToWidth(image_width, Qt.SmoothTransformation)
                            return QIcon(thumb_pixmap)
                        else:
                            return None #Explicitly return None if the pixmap fails.
            return None  # Return None if any of the shell operations fail

        except ImportError:
            print("pywin32 is not installed. OS thumbnail retrieval on Windows will not work.")
            return None
        except Exception as e:
            print(f"Error getting OS thumbnail: {e}")
            return None
    elif sys.platform == "darwin":
        try:
            # macOS QuickLook (qlmanage)
            # Use qlmanage to generate the thumbnail
            command = [
                "qlmanage",
                "-t", str(file_path),  # -t generates thumbnail
                "-s", f"{image_width}x{image_width}",  # Set the size
                "-o", "/tmp",  # Output directory
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            # qlmanage creates a file named <original_name>.png in /tmp
            thumbnail_path = os.path.join("/tmp", os.path.basename(file_path) + ".png")

            if os.path.exists(thumbnail_path):
                pixmap = QPixmap(thumbnail_path)
                if not pixmap.isNull():
                    # qlmanage sometimes ignores the size request, so we need to scale
                    pixmap = pixmap.scaledToWidth(image_width, Qt.SmoothTransformation)
                    os.remove(thumbnail_path)  # Clean up the temporary file
                    return QIcon(pixmap)
                else:
                    os.remove(thumbnail_path)
                    return None
            else:
                return None

        except subprocess.CalledProcessError as e:
            print(f"qlmanage failed: {e}")
            return None
        except Exception as e:
            print(f"Error getting macOS thumbnail: {e}")
            return None


    elif sys.platform.startswith("linux"):  #linux (and other unix-like OS's)
        try:
            #Freedesktop Thumbnail Standard.
            # https://specifications.freedesktop.org/thumbnail-spec/thumbnail-spec-latest.html

            # 1. Get the URI
            import urllib.parse
            uri = "file://" + urllib.parse.quote(os.path.abspath(file_path))

            # 2. Get mtime
            mtime = int(os.path.getmtime(file_path))

            # 3.  Hash the URI
            import hashlib
            hash = hashlib.md5(uri.encode('utf-8')).hexdigest()

            # 4.  Look for the thumbnail in the cache locations.
            thumbnail_dir = os.path.join(os.path.expanduser("~"), ".cache", "thumbnails")
            #Normal Size
            normal_path = os.path.join(thumbnail_dir, "normal", f"{hash}.png")
            #Large Size
            large_path = os.path.join(thumbnail_dir, "large", f"{hash}.png")

            thumb_path_to_use = None
            #First check large
            if os.path.exists(large_path):
                #check mtime.
                thumb_mtime = int(os.path.getmtime(large_path))
                if thumb_mtime >= mtime: # valid thumbnail
                    thumb_path_to_use = large_path
            #else check normal:
            if thumb_path_to_use is None and os.path.exists(normal_path):
                thumb_mtime = int(os.path.getmtime(normal_path))
                if thumb_mtime >= mtime:  #valid thumbnail
                    thumb_path_to_use = normal_path
            #We found a valid cache
            if thumb_path_to_use:
                #Check the thumbnail actually loads!
                pixmap = QPixmap(thumb_path_to_use)
                if not pixmap.isNull():
                    pixmap = pixmap.scaledToWidth(image_width, Qt.SmoothTransformation)
                    return QIcon(pixmap)
                else:
                    #Remove bad thumbnail
                    try:
                        os.remove(thumb_path_to_use)
                    except:
                        pass
                    return None
            return None

        except Exception as e:
            print(f"Error getting Linux thumbnail: {e}")
            return None
    else: #other operating system
        return None

def get_file_paths(directory_path: Path) -> set[Path]:
    """
    Recursively get all file paths in a directory, including 
    subdirectories.
    """
    file_paths = set()
    for path in directory_path.rglob("*"):  # Use rglob for recursive search
        if path.is_file():
            file_paths.add(path)
    return file_paths


@dataclass
class HistoryItem:
    action_name: str
    tags: list[list[str]]
    should_ask_for_confirmation: bool


class Scope(str, Enum):
    ALL_IMAGES = 'All images'
    FILTERED_IMAGES = 'Filtered images'
    SELECTED_IMAGES = 'Selected images'

class ImageListModel(QAbstractListModel):
    update_undo_and_redo_actions_requested = Signal()

    def __init__(self, image_list_image_width: int, tag_separator: str):
        super().__init__()
        self.image_list_image_width = image_list_image_width
        self.tag_separator = tag_separator
        self.images: list[Image] = []
        self.undo_stack = deque(maxlen=UNDO_STACK_SIZE)
        self.redo_stack = []
        self.proxy_image_list_model = None
        self.image_list_selection_model = None
        self.time_stats = Counter()


    def rowCount(self, parent=None) -> int:
        return len(self.images)

    def data(self, index: QModelIndex, role=None) -> Image | str | QIcon | QSize:
        if not index.isValid():
            return None  # Important: handle invalid indices

        image = self.images[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return image
        if role == Qt.ItemDataRole.DisplayRole:
            text = image.path.name
            if image.tags:
                caption = self.tag_separator.join(image.tags)
                text += f'\n{caption}'
            return text
        if role == Qt.ItemDataRole.DecorationRole:
            if image.thumbnail:
                return image.thumbnail
            try:
                return self.get_icon(image, self.image_list_image_width)
            except Exception as e:
                print(f"Error getting icon for {image.path} {e}")
                return QIcon()

        if role == Qt.ItemDataRole.SizeHintRole:
            if image.thumbnail:
                return image.thumbnail.availableSizes()[0]
            dimensions = image.dimensions
            if not dimensions:
                return QSize(self.image_list_image_width,
                             self.image_list_image_width)
            width, height = dimensions
            # Scale the dimensions to the image width.
            return QSize(self.image_list_image_width,
                         int(self.image_list_image_width * height / width))


    def get_icon(self, image, image_width: int) -> QIcon:
        """Load the image and return a QIcon while keeping aspect ratio."""
        try:
            # --- OS Thumbnail Cache ---
            start = time.perf_counter()
            thumbnail = get_os_thumbnail(image.path, image_width)
            self.time_stats['os_thumbnail_check'] += time.perf_counter() - start
            if thumbnail:
                image.thumbnail = thumbnail  # Store the thumbnail
                return thumbnail

            # --- JXL Handling ---
            if image.path.suffix.lower() == ".jxl":
                start = time.perf_counter()
                try:
                    # Attempt to use Qt's built-in JXL support (might require plugin)
                    qimage = QImage(str(image.path))
                    if not qimage.isNull():
                        pixmap = QPixmap.fromImage(qimage)
                        pixmap = pixmap.scaled(image_width, image_width * 3, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        icon = QIcon(pixmap)
                        self.time_stats['qt_jxl_path'] += time.perf_counter() - start
                        image.thumbnail = icon
                        return icon
                    else:  # Fallback to pillow-jxl if Qt fails
                        raise Exception("Qt couldn't load JXL directly")


                except Exception as e:  # Qt JXL support may not be available
                    self.time_stats['jxl_thumbnail_to_qicon'] += time.perf_counter() - start
                    print(f'failed getting jxl thumbnail due to {e}')
                    start = time.perf_counter()
                    pil_image = pilimage.open(image.path)  # Uses pillow-jxl
                    qimage = pil_to_qimage(pil_image)

                    # Convert to QPixmap and scale while keeping aspect ratio
                    pixmap = QPixmap.fromImage(qimage)
                    pixmap = pixmap.scaled(image_width, image_width * 3, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    icon = QIcon(pixmap)
                    self.time_stats['jxl_exception_path'] += time.perf_counter() - start
                    image.thumbnail = icon
                    return icon

            # --- Standard Image Handling ---
            else:
                start = time.perf_counter()
                image_reader = QImageReader(str(image.path))
                # Rotate the image based on the orientation tag.
                image_reader.setAutoTransform(True)

                # Try reading just the scaled thumbnail directly (optimization)
                image_reader.setScaledSize(QSize(image_width, int(image_width * 1.5))) #Estimate height for aspect
                qimage = image_reader.read() #Directly try the read.
                if qimage.isNull():
                    # If direct scaled read fails, load the full image
                    image_reader = QImageReader(str(image.path))
                    image_reader.setAutoTransform(True)
                    pixmap = QPixmap.fromImageReader(image_reader).scaledToWidth(
                        image_width, Qt.TransformationMode.SmoothTransformation
                        )
                else:
                    pixmap = QPixmap.fromImage(qimage)  # Use qimage

                thumbnail = QIcon(pixmap)
                image.thumbnail = thumbnail
                self.time_stats['non_jxl_path'] += time.perf_counter() - start
                return thumbnail

        except Exception as e:
            print(f"Error loading image {image.path}: {e}")
            return QIcon()       


    def load_directory(self, directory_path: Path):
        self.images.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_and_redo_actions_requested.emit()
        file_paths = get_file_paths(directory_path)
        settings = get_settings()
        image_suffixes_string = settings.value(
            'image_list_file_formats',
            defaultValue=DEFAULT_SETTINGS['image_list_file_formats'], type=str)
        image_suffixes = []
        for suffix in image_suffixes_string.split(','):
            suffix = suffix.strip().lower()
            if not suffix.startswith('.'):
                suffix = '.' + suffix
            image_suffixes.append(suffix)
        image_paths = {path for path in file_paths
                       if path.suffix.lower() in image_suffixes}
        # Comparing paths is slow on some systems, so convert the paths to
        # strings.
        text_file_path_strings = {str(path) for path in file_paths
                                  if path.suffix == '.txt'}
        for image_path in image_paths:
            try:
                if str(image_path).endswith('jxl'):
                    dimensions = get_jxl_size(image_path)
                else:
                    dimensions = pilimage.open(image_path).size
                with open(image_path, 'rb') as image_file:
                    try:
                        exif_tags = exifread.process_file(
                            image_file, details=False,
                            stop_tag='Image Orientation')
                        if 'Image Orientation' in exif_tags:
                            orientations = (exif_tags['Image Orientation']
                                            .values)
                            if any(value in orientations
                                for value in (5, 6, 7, 8)):
                                dimensions = (dimensions[1], dimensions[0])
                    except Exception as exception:
                        print(f'Failed to get Exif tags for {image_path}: '
                            f'{exception}', file=sys.stderr)
            except (ValueError, OSError) as exception:
                print(f'Failed to get dimensions for {image_path}: '
                      f'{exception}', file=sys.stderr)
                dimensions = None
            
            tags = []
            text_file_path = image_path.with_suffix('.txt')
            if str(text_file_path) in text_file_path_strings:
                # `errors='replace'` inserts a replacement marker such as '?'
                # when there is malformed data.
                caption = text_file_path.read_text(encoding='utf-8',
                                                   errors='replace')
                if caption:
                    tags = caption.split(self.tag_separator)
                    tags = [tag.strip() for tag in tags]
                    tags = [tag for tag in tags if tag]
            image = Image(image_path, dimensions, tags)
            self.images.append(image)
            
        self.images.sort(key=lambda image_: image_.path)
        self.modelReset.emit()

    def add_to_undo_stack(self, action_name: str,
                          should_ask_for_confirmation: bool):
        """Add the current state of the image tags to the undo stack."""
        tags = [image.tags.copy() for image in self.images]
        self.undo_stack.append(HistoryItem(action_name, tags,
                                           should_ask_for_confirmation))
        self.redo_stack.clear()
        self.update_undo_and_redo_actions_requested.emit()

    def write_image_tags_to_disk(self, image: Image):
        try:
            image.path.with_suffix('.txt').write_text(
                self.tag_separator.join(image.tags), encoding='utf-8',
                errors='replace')
        except OSError:
            QMessageBox.critical(None, 'Error',
                                 f'Failed to save tags for {image.path}.')  # Simpler error message

    def restore_history_tags(self, is_undo: bool):
        if is_undo:
            source_stack = self.undo_stack
            destination_stack = self.redo_stack
        else:
            # Redo.
            source_stack = self.redo_stack
            destination_stack = self.undo_stack
        if not source_stack:
            return
        history_item = source_stack[-1]
        if history_item.should_ask_for_confirmation:
            undo_or_redo_string = 'Undo' if is_undo else 'Redo'
            reply = get_confirmation_dialog_reply(
                title=undo_or_redo_string,
                question=f'{undo_or_redo_string} '
                         f'"{history_item.action_name}"?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        source_stack.pop()
        tags = [image.tags for image in self.images]
        destination_stack.append(HistoryItem(
            history_item.action_name, tags,
            history_item.should_ask_for_confirmation))
        
        changed_image_indices = []
        for image_index, (image, history_image_tags) in enumerate(
                zip(self.images, history_item.tags)):
            if image.tags == history_image_tags:
                continue
            changed_image_indices.append(image_index)
            image.tags = history_image_tags
            self.write_image_tags_to_disk(image)

        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
        self.update_undo_and_redo_actions_requested.emit()

    @Slot()
    def undo(self):
        """Undo the last action."""
        self.restore_history_tags(is_undo=True)

    @Slot()
    def redo(self):
        """Redo the last undone action."""
        self.restore_history_tags(is_undo=False)

    def is_image_in_scope(self, scope: Scope | str, image_index: int,
                          image: Image) -> bool:
        if scope == Scope.ALL_IMAGES:
            return True
        if scope == Scope.FILTERED_IMAGES:
            return self.proxy_image_list_model.is_image_in_filtered_images(
                image)
        if scope == Scope.SELECTED_IMAGES:
            proxy_index = self.proxy_image_list_model.mapFromSource(
                self.index(image_index))
            return self.image_list_selection_model.isSelected(proxy_index)

    def get_text_match_count(self, text: str, scope: Scope | str,
                             whole_tags_only: bool, use_regex: bool) -> int:
        """Get the number of instances of a text in all captions."""
        match_count = 0
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if whole_tags_only:
                if use_regex:
                    match_count += len([
                        tag for tag in image.tags
                        if re.fullmatch(pattern=text, string=tag)
                    ])
                else:
                    match_count += image.tags.count(text)
            else:
                caption = self.tag_separator.join(image.tags)
                if use_regex:
                    match_count += len(re.findall(pattern=text,
                                                  string=caption))
                else:
                    match_count += caption.count(text)
        return match_count

    def find_and_replace(self, find_text: str, replace_text: str,
                         scope: Scope | str, use_regex: bool):
        """
        Find and replace arbitrary text in captions, within and across tag
        boundaries.
        """
        if not find_text:
            return
        self.add_to_undo_stack(action_name='Find and Replace',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            caption = self.tag_separator.join(image.tags)
            if use_regex:
                if not re.search(pattern=find_text, string=caption):
                    continue
                caption = re.sub(pattern=find_text, repl=replace_text,
                                 string=caption)
            else:
                if find_text not in caption:
                    continue
                caption = caption.replace(find_text, replace_text)
            changed_image_indices.append(image_index)
            image.tags = caption.split(self.tag_separator)
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def sort_tags_alphabetically(self, do_not_reorder_first_tag: bool):
        """Sort the tags for each image in alphabetical order."""
        self.add_to_undo_stack(action_name='Sort Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_caption = self.tag_separator.join(image.tags)
            if do_not_reorder_first_tag:
                first_tag = image.tags[0]
                image.tags = [first_tag] + sorted(image.tags[1:])
            else:
                image.tags.sort()
            new_caption = self.tag_separator.join(image.tags)
            if new_caption != old_caption:
                changed_image_indices.append(image_index)
                self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def sort_tags_by_frequency(self, tag_counter: Counter,
                               do_not_reorder_first_tag: bool):
        """
        Sort the tags for each image by the total number of times a tag appears
        across all images.
        """
        self.add_to_undo_stack(action_name='Sort Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_caption = self.tag_separator.join(image.tags)
            if do_not_reorder_first_tag:
                first_tag = image.tags[0]
                image.tags = [first_tag] + sorted(
                    image.tags[1:], key=lambda tag: tag_counter[tag],
                    reverse=True)
            else:
                image.tags.sort(key=lambda tag: tag_counter[tag], reverse=True)
            new_caption = self.tag_separator.join(image.tags)
            if new_caption != old_caption:
                changed_image_indices.append(image_index)
                self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def reverse_tags_order(self, do_not_reorder_first_tag: bool):
        """Reverse the order of the tags for each image."""
        self.add_to_undo_stack(action_name='Reverse Order of Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            changed_image_indices.append(image_index)
            if do_not_reorder_first_tag:
                image.tags = [image.tags[0]] + list(reversed(image.tags[1:]))
            else:
                image.tags = list(reversed(image.tags))
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def shuffle_tags(self, do_not_reorder_first_tag: bool):
        """Shuffle the tags for each image randomly."""
        self.add_to_undo_stack(action_name='Shuffle Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            changed_image_indices.append(image_index)
            if do_not_reorder_first_tag:
                first_tag, *remaining_tags = image.tags
                random.shuffle(remaining_tags)
                image.tags = [first_tag] + remaining_tags
            else:
                random.shuffle(image.tags)
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def move_tags_to_front(self, tags_to_move: list[str]):
        """
        Move one or more tags to the front of the tags list for each image.
        """
        self.add_to_undo_stack(action_name='Move Tags to Front',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if not any(tag in image.tags for tag in tags_to_move):
                continue
            old_caption = self.tag_separator.join(image.tags)
            moved_tags = []
            for tag in tags_to_move:
                tag_count = image.tags.count(tag)
                moved_tags.extend([tag] * tag_count)
            unmoved_tags = [tag for tag in image.tags if tag not in moved_tags]
            image.tags = moved_tags + unmoved_tags
            new_caption = self.tag_separator.join(image.tags)
            if new_caption != old_caption:
                changed_image_indices.append(image_index)
                self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def remove_duplicate_tags(self) -> int:
        """
        Remove duplicate tags for each image. Return the number of removed
        tags.
        """
        self.add_to_undo_stack(action_name='Remove Duplicate Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        removed_tag_count = 0
        for image_index, image in enumerate(self.images):
            tag_count = len(image.tags)
            unique_tag_count = len(set(image.tags))
            if tag_count == unique_tag_count:
                continue
            changed_image_indices.append(image_index)
            removed_tag_count += tag_count - unique_tag_count
            image.tags = list(dict.fromkeys(image.tags))
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
        return removed_tag_count

    def remove_empty_tags(self) -> int:
        """
        Remove empty tags (tags that are empty strings or only contain
        whitespace) for each image. Return the number of removed tags.
        """
        self.add_to_undo_stack(action_name='Remove Empty Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        removed_tag_count = 0
        for image_index, image in enumerate(self.images):
            old_tag_count = len(image.tags)
            image.tags = [tag for tag in image.tags if tag.strip()]
            new_tag_count = len(image.tags)
            if old_tag_count == new_tag_count:
                continue
            changed_image_indices.append(image_index)
            removed_tag_count += old_tag_count - new_tag_count
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
        return removed_tag_count

    def update_image_tags(self, image_index: QModelIndex, tags: list[str]):
        image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
        if image.tags == tags:
            return
        image.tags = tags
        self.dataChanged.emit(image_index, image_index)
        self.write_image_tags_to_disk(image)


    @Slot(list, list)
    def add_tags(self, tags: list[str], image_indices: list[QModelIndex]):
        """Add one or more tags to one or more images."""
        if not image_indices:
            return
        action_name = f'Add {pluralize("Tag", len(tags))}'
        should_ask_for_confirmation = len(image_indices) > 1
        self.add_to_undo_stack(action_name, should_ask_for_confirmation)
        for image_index in image_indices:
            image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
            image.tags.extend(tags)
            self.write_image_tags_to_disk(image)
        min_image_index = min(image_indices, key=lambda index: index.row())
        max_image_index = max(image_indices, key=lambda index: index.row())
        self.dataChanged.emit(min_image_index, max_image_index)


    @Slot(list, str)
    def rename_tags(self, old_tags: list[str], new_tag: str,
                    scope: Scope | str = Scope.ALL_IMAGES,
                    use_regex: bool = False):
        self.add_to_undo_stack(
            action_name=f'Rename {pluralize("Tag", len(old_tags))}',
            should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if use_regex:
                pattern = old_tags[0]
                if not any(re.fullmatch(pattern=pattern, string=image_tag)
                           for image_tag in image.tags):
                    continue
                image.tags = [new_tag if re.fullmatch(pattern=pattern,
                                                      string=image_tag)
                              else image_tag for image_tag in image.tags]
            else:
                if not any(old_tag in image.tags for old_tag in old_tags):
                    continue
                image.tags = [new_tag if image_tag in old_tags else image_tag
                              for image_tag in image.tags]
            changed_image_indices.append(image_index)
            self.write_image_tags_to_disk(image)

        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    @Slot(list)
    def delete_tags(self, tags: list[str],
                    scope: Scope | str = Scope.ALL_IMAGES,
                    use_regex: bool = False):
        self.add_to_undo_stack(
            action_name=f'Delete {pluralize("Tag", len(tags))}',
            should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            
            if use_regex:
                pattern = tags[0]
                if not any(re.fullmatch(pattern=pattern, string=image_tag)
                           for image_tag in image.tags):
                    continue
                image.tags = [image_tag for image_tag in image.tags
                              if not re.fullmatch(pattern=pattern,
                                                  string=image_tag)]
            else:
                if not any(tag in image.tags for tag in tags):
                    continue
                image.tags = [image_tag for image_tag in image.tags
                              if image_tag not in tags]
            changed_image_indices.append(image_index)
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
