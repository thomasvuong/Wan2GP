import os
import base64
import io
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    import comtypes
    from comtypes import IUnknown, GUID, HRESULT, COMMETHOD

    class SIZE(ctypes.Structure):
        _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]
    class IShellItemImageFactory(IUnknown):
        _case_insensitive_ = True
        _iid_ = GUID('{bcc18b79-ba16-442f-80c4-8a59c30c463b}')
        _idlflags_ = []
        _methods_ = [COMMETHOD([], HRESULT, 'GetImage', (['in'], SIZE, 'size'), (['in'], ctypes.c_uint, 'flags'), (['out', 'retval'], ctypes.POINTER(wintypes.HBITMAP), 'phbm'))]
    SIIGBF_THUMBNAILONLY = 0x8
    shell32 = ctypes.windll.shell32
    gdi32 = ctypes.windll.gdi32
    SHCreateItemFromParsingName = shell32.SHCreateItemFromParsingName
    SHCreateItemFromParsingName.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.POINTER(IShellItemImageFactory))]
    SHCreateItemFromParsingName.restype = HRESULT
    class BITMAP(ctypes.Structure):
        _fields_ = [("bmType", wintypes.LONG), ("bmWidth", wintypes.LONG), ("bmHeight", wintypes.LONG), ("bmWidthBytes", wintypes.LONG), ("bmPlanes", wintypes.WORD), ("bmBitsPixel", wintypes.WORD), ("bmBits", wintypes.LPVOID)]
    GetObjectW = gdi32.GetObjectW
    GetObjectW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(BITMAP)]
    GetObjectW.restype = ctypes.c_int
    DeleteObject = gdi32.DeleteObject
    DeleteObject.argtypes = [ctypes.c_void_p]
    DeleteObject.restype = wintypes.BOOL

    def get_thumbnail_as_base64(file_path, size=256):
        comtypes.CoInitialize()
        factory_ptr = ctypes.POINTER(IShellItemImageFactory)()
        hbitmap_handle = 0
        bmp_copy = None
        try:
            hr = SHCreateItemFromParsingName(file_path, None, IShellItemImageFactory._iid_, ctypes.byref(factory_ptr))
            if hr != 0: return None, file_path
            size_struct = SIZE(size, size)
            hbitmap_handle = factory_ptr.GetImage(size_struct, SIIGBF_THUMBNAILONLY)
            if hbitmap_handle:
                try:
                    bitmap_info = BITMAP()
                    if GetObjectW(hbitmap_handle, ctypes.sizeof(bitmap_info), ctypes.byref(bitmap_info)) == 0: return None, file_path
                    bmp = Image.frombuffer('RGB', (bitmap_info.bmWidth, bitmap_info.bmHeight), ctypes.string_at(bitmap_info.bmBits, bitmap_info.bmWidthBytes * bitmap_info.bmHeight), 'raw', 'BGRX', bitmap_info.bmWidthBytes, -1)
                    bmp_copy = bmp.transpose(Image.FLIP_TOP_BOTTOM)
                finally:
                    DeleteObject(hbitmap_handle)
        except comtypes.COMError:
            return None, file_path
        except Exception as e:
            print(f"Error extracting native thumbnail for {os.path.basename(file_path)}: {e}")
            return None, file_path
        finally:
            if factory_ptr: del factory_ptr
            comtypes.CoUninitialize()
        if bmp_copy:
            try:
                buffer = io.BytesIO()
                bmp_copy.save(buffer, format="JPEG", quality=80)
                return base64.b64encode(buffer.getvalue()).decode('utf-8'), file_path
            except Exception as e:
                print(f"Failed to convert or save image for '{os.path.basename(file_path)}': {e}")
        return None, file_path
else:
    def get_thumbnail_as_base64(file_path, size=256):
        return None, file_path

def get_thumbnails_in_batch_windows(file_paths):
    if not file_paths or os.name != 'nt':
        return {}
    results = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for thumbnail, path in executor.map(get_thumbnail_as_base64, file_paths):
            if thumbnail: results[path] = thumbnail
    return results