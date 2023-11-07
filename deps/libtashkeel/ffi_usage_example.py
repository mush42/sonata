# coding: utf-8

import ctypes
import os


# Change this based on library path
LIBTASHKEEL_PATH = os.path.abspath("../target/debug/libtashkeel.dll")

class LibtashkeelError(ctypes.Structure):
    _fields_ = [
        ("err_code", ctypes.c_int32),
        ("err_msg_ptr", ctypes.c_void_p),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message = None

    @property
    def message(self):
        if self.err_msg_ptr:
            self._message = ctypes.cast(self.err_msg_ptr, ctypes.c_char_p).value.decode('utf-8')
            self.__free_err_msg_string()

    def __del__(self):
        try:
            self.__free_err_msg_string()
        except:
            pass

    def __free_err_msg_string(self):
        lib.libtashkeel_free_string(self.err_msg_ptr)


lib = ctypes.cdll.LoadLibrary(LIBTASHKEEL_PATH)

lib.libtashkeelTashkeel.argtypes = (ctypes.c_char_p, LibtashkeelError)
lib.libtashkeelTashkeel.restype = ctypes.c_void_p
lib.libtashkeel_free_string.argtypes = (ctypes.c_void_p, )


def tashkeel(text):
    err = LibtashkeelError()
    ptr = lib.libtashkeelTashkeel(
        ctypes.c_char_p(text.encode("utf-8")),
        err
    )
    if err.err_code != 0:
        raise RuntimeError(err.message)
    try:
        res = ctypes.cast(ptr, ctypes.c_char_p).value.decode('utf-8')
    finally:
        lib.libtashkeel_free_string(ptr)
    return res

