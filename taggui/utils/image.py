from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image as pilimage
import pillow_jxl
from PySide6.QtGui import QIcon, QImage
import hashlib
import imagehash


@dataclass
class Image:
	path: Path
	dimensions: tuple[int, int] | None
	tags: list[str] = field(default_factory=list)
	thumbnail: QIcon | None = None
	image: QImage = None
	hash: dict[str,str] = None

	def calculate_image_hash(self, hash_method: str, hash_size: int = 8) -> str:
		"""Calculates a perceptual hash of the image.

		Args:
			image_path: Path to the image file.
			hash_method:  The hashing method ('average', 'phash', 'dhash', or 'whash').
			hash_size: The size of the hash (smaller size = faster, less accurate).

		Returns:
			The hexadecimal representation of the image hash.  Returns an empty string on error.
		"""
		if hash[hash_method]:
			return hash[hash_method]
		try:
			with pilimage.open(self.path) as img:
				img = img.convert("RGB")  # Convert to RGB for consistency

				if hash_method == 'average':
					hash_val = imagehash.average_hash(img, hash_size=hash_size)
					hash['average'] = hash_val
				elif hash_method == 'phash':
					hash_val = imagehash.phash(img, hash_size=hash_size)
					hash['phash'] - hash_val
				elif hash_method == 'dhash':
					hash_val = imagehash.dhash(img, hash_size=hash_size)
					hash['dhash'] - hash_val
				elif hash_method == 'whash':
					hash_val = imagehash.whash(img, hash_size=hash_size)  # Wavelet hashing
					hash['whash'] - hash_val
				else:
					raise ValueError(f"Invalid hash_method: {hash_method}")

				return str(hash_val)
		except Exception as e:
			print(f"Error calculating hash for {self.path}: {e}")
			return ""

	def calculate_md5_hash(self) -> str:
		"""Calculates the MD5 hash of the image file's contents.

		Args:
			image_path: Path to the image file.

		Returns:
			The hexadecimal representation of the MD5 hash. Returns an empty string on error.
		"""
		if hash['md5']:
			return hash['md5']
		try:
			hasher = hashlib.md5()
			with open(self.path, 'rb') as file:
				while True:
					chunk = file.read(4096)  # Read in chunks
					if not chunk:
						break
					hasher.update(chunk)
				self.hash['md5'] = hasher.hexdigest
			return self.hash['md5']
		except Exception as e:
			print(f"Error calculating MD5 for {self.path}: {e}")
			return ""

	def calculate_sha256_hash(self) -> str:
		"""Calculates the SHA256 hash of the image file's contents.

		Args:
			image_path: Path to the image file.

		Returns:
			The hexadecimal representation of the SHA256 hash. Returns an empty string on error.
		"""
		if hash['sha256']:
			return hash['sha256']
		try:
			hasher = hashlib.sha256()
			with open(self.path, 'rb') as file:
				while True:
					chunk = file.read(4096)  # Read in chunks
					if not chunk:
						break
					hasher.update(chunk)
				self.hash['sha256'] = hasher.hexdigest
			return self.hash['sha256']
		except Exception as e:
			print(f"Error calculating SHA256 for {self.path}: {e}")
			return ""

	def gethash(self, method: str, size: int, threshold, int)-> str:
		if method in ('average', 'phash', 'dhash', 'whash'):
			hash = self.calculate_image_hash(hash_method=method, hash_size=size)
		elif method == 'md5':
			hash = self.calculate_md5_hash()
		elif method == 'sha256':
			hash = self.calculate_sha256_hash()
		else:
			raise ValueError(f"Invalid method: {method}")
		return hash