"""Companion — local AI voice companion."""

__version__ = "0.1.0"

# Suppress ALSA error messages on Linux/WSL before audio libraries initialize.
import ctypes as _ctypes

try:
    _ERROR_HANDLER_FUNC = _ctypes.CFUNCTYPE(
        None,
        _ctypes.c_char_p,
        _ctypes.c_int,
        _ctypes.c_char_p,
        _ctypes.c_int,
        _ctypes.c_char_p,
    )

    def _null_error_handler(filename, line, function, err, fmt):
        pass

    _c_error_handler = _ERROR_HANDLER_FUNC(_null_error_handler)
    _asound = _ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_c_error_handler)
except OSError:
    pass
