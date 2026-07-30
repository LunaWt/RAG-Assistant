import os
import tempfile

dir_object = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

os.environ['STORAGE_DIR'] = dir_object.name