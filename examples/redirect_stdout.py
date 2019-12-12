"""
Functions in this file allow redirecting stdout used by Fortran and C
extensions to a different file or devnull or stderr.

Example usage (same should work with stdout_redirect_1()):

# No print to stdout:
with stdout_redirect_2():
    call_fortran_or_c_code()
# OR:
with stdout_redirect_1():
    call_fortran_or_c_code()

# Get access to what was meant to be printed:
import io
f = io.BytesIO()
with stdout_redirect_2(f):
    call_fortran_or_c_code()
print('Got stdout: "{0}"'.format(f.getvalue().decode('utf-8')))
"""
import os
import sys
import ctypes
import io
import tempfile
from contextlib import contextmanager


# Code based on:
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/17954769#17954769
@contextmanager
def stdout_redirect_1(to=os.devnull):
    """
    Parameter *to* is str type and indicates file name.
    """
    fd = sys.stdout.fileno()
    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file_:
            _redirect_stdout(to=file_)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as CLOEXEC may be different


# Code based on:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

@contextmanager
def stdout_redirect_2(to=None):
    """
    Parameter *to* is steam or similar type.
    """
    if to is None:
        to = io.BytesIO()
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        to.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)
