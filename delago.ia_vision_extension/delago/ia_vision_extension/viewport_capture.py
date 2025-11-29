# viewport_capture.py
import os
import time
import asyncio
import omni.kit.viewport.utility as vp_utils
from omni.kit.widget.viewport.capture import FileCapture
import omni.kit.pipapi

omni.kit.pipapi.install("PIllow", ignore_import_check=True)

from PIL import Image


class ViewportCapturer:

    """
    Utility to capture the active viewport to a PNG file.
    """

    def __init__(self):

        """
        Prepare the output directory for captures.
        """

        ext_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_folder = os.path.join(ext_root, "data")
        os.makedirs(self.data_folder, exist_ok=True)


    async def save_async(self) -> str | None:

        """Capture the viewport to disk and wait for a stable file size.

        Args:
            None.

        Returns:
            Absolute file path to the PNG on success; None on failure.
        """

        vw = vp_utils.get_active_viewport_window()
        if vw is None:
            return None

        # Unique per run: timestamp-only filename (e.g., viewport_1725961234567.png)
        # This avoids that old pictures get taken by LLM, because new ones where too slow
        fname = f"viewport_{int(time.time() * 1000)}.png"
        file_path = os.path.join(self.data_folder, fname)

        capture = vw.viewport_api.schedule_capture(FileCapture(file_path))
        result = await capture.wait_for_result()
        if not result:
            return None

        # Wait for size to stabilize (avoids partial/old reads)
        last = -1
        for _ in range(40):  # max ~2s total (40*0.05)
            try:
                size = os.path.getsize(file_path)
                if size > 0 and size == last:
                    with open(file_path, "rb"):
                        pass

                    # --- Convert to JPEG inline ---
                    jpg_path = file_path.replace(".png", ".jpg")
                    im = Image.open(file_path).convert("RGB")
                    w, h = im.size
                    im = im.resize((int(w * 0.816), int(h * 0.816)), Image.LANCZOS)
                    im.save(jpg_path, "JPEG", quality=90, optimize=True, progressive=True)

                    os.remove(file_path)  # remove original PNG
                    return jpg_path
                last = size
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
            await asyncio.sleep(0.05)

        return None
