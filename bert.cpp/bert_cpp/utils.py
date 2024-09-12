# various utils (from llama-cpp-python mostly)

import os
import sys
import ctypes
from pathlib import Path

# load the library
def load_shared_library(lib_base_name):
    # construct the paths to the possible shared library names
    base_path = Path(os.path.abspath(os.path.dirname(__file__)))

    # searching for the library in the current directory under the name 'libbert' (default name
    # for bert.cpp) and 'bert' (default name for this repo)
    lib_paths = []

    # Determine the file extension based on the platform
    if sys.platform.startswith('linux'):
        lib_paths += [
            base_path / f'lib{lib_base_name}.so',
        ]
    elif sys.platform == 'darwin':
        lib_paths += [
            base_path / f'lib{lib_base_name}.so',
            base_path / f'lib{lib_base_name}.dylib',
        ]
    elif sys.platform == 'win32':
        lib_paths += [
            base_path / f'{lib_base_name}.dll',
            base_path / f'lib{lib_base_name}.dll',
        ]
    else:
        raise RuntimeError('Unsupported platform')

    if 'BERT_CPP_LIB' in os.environ:
        lib_path = Path(os.environ['BERT_CPP_LIB'])
        base_path = lib_path.parent.resolve()
        lib_paths = [lib_path.resolve()]

    # add the library directory to the DLL search path on Windows (if needed)
    cdll_args = dict()
    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        os.add_dll_directory(str(base_path))
        if 'CUDA_PATH' in os.environ:
            os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
            os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'lib'))
        if 'HIP_PATH' in os.environ:
            os.add_dll_directory(os.path.join(os.environ['HIP_PATH'], 'bin'))
            os.add_dll_directory(os.path.join(os.environ['HIP_PATH'], 'lib'))
        cdll_args['winmode'] = ctypes.RTLD_GLOBAL

    # try to load the shared library, handling potential errors
    for lib_path in lib_paths:
        if lib_path.exists():
            try:
                return ctypes.CDLL(str(lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f'Failed to load shared library "{lib_path}": {e}')

    raise FileNotFoundError(
        f'Shared library with base name "{lib_base_name}" not found'
    )

# avoid 'LookupError: unknown encoding: ascii' when open() called in a destructor
outnull_file = open(os.devnull, 'w')
errnull_file = open(os.devnull, 'w')

class suppress_stdout_stderr():
    # NOTE: these must be 'saved' here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os

    def __init__(self, disable=True):
        self.disable = disable

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        if self.disable:
            return self

        # Check if sys.stdout and sys.stderr have fileno method
        if not hasattr(self.sys.stdout, 'fileno') or not hasattr(self.sys.stderr, 'fileno'):
            return self  # Return the instance without making changes

        self.old_stdout_fileno_undup = self.sys.stdout.fileno()
        self.old_stderr_fileno_undup = self.sys.stderr.fileno()

        self.old_stdout_fileno = self.os.dup(self.old_stdout_fileno_undup)
        self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)

        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.os.dup2(outnull_file.fileno(), self.old_stdout_fileno_undup)
        self.os.dup2(errnull_file.fileno(), self.old_stderr_fileno_undup)

        self.sys.stdout = outnull_file
        self.sys.stderr = errnull_file
        return self

    def __exit__(self, *_):
        if self.disable:
            return

        # Check if sys.stdout and sys.stderr have fileno method
        if hasattr(self.sys.stdout, 'fileno') and hasattr(self.sys.stderr, 'fileno'):
            self.sys.stdout = self.old_stdout
            self.sys.stderr = self.old_stderr

            self.os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
            self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

            self.os.close(self.old_stdout_fileno)
            self.os.close(self.old_stderr_fileno)
