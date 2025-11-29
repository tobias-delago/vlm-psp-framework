import omni.ext
import omni.ui as ui
import asyncio
import os
import carb
import csv
from .camera_control import CameraControl
from .main import ViewportAnalyzer
from .viewport_capture import ViewportCapturer

from .hugging_face import call_hugging_face

ROW_H = 26; LBL_NARROW_W = 18; LBL_W = 80; FIELD_W = 80; FIELD_H = 22

# helper function for drop down in settings
def simple_combo(items, default_idx=0):
    class _Item(ui.AbstractItem):
        def __init__(self, text): super().__init__(); self.model = ui.SimpleStringModel(text)

    class _Model(ui.AbstractItemModel):
        def __init__(self):
            super().__init__()
            self.index = ui.SimpleIntModel(default_idx)
            self.index.add_value_changed_fn(lambda *_: self._item_changed(None))
            self.items = [_Item(t) for t in items]
        def get_item_children(self, item): return self.items
        def get_item_value_model(self, item, col):
            return self.index if item is None else item.model

    return _Model()

class MyExtension(omni.ext.IExt):
    """
    Minimal Omniverse extension that shows a main panel with a header
    and a settings icon. The settings window lets the user enter:
      - URL
      - API Key
      - Model
      - Mode
    and saves them persistently to carb settings under /persistent/ia_vision/.
    """

    # Carb settings keys (single source of truth)
    S_URL   = "/persistent/ia_vision/url"
    S_KEY   = "/persistent/ia_vision/api_key"
    S_MODEL = "/persistent/ia_vision/model"
    S_MODE  = "/persistent/ia_vision/mode"


    def on_startup(self, _ext_id):
        # ---------------------------------------------------------------------
        # Paths
        # Resolve the /data folder relative to this file (no hardcoded absolute paths)
        ext_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_folder = os.path.join(ext_root, "data")
        os.makedirs(self.data_folder, exist_ok=True)
        self._settings_icon_path = os.path.join(self.data_folder, "settings_icon.png")

        # ---------------------------------------------------------------------
        # Carb settings handle
        self._settings = carb.settings.get_settings()

        # UI models pre-filled from persistent settings
        self._url_model        = ui.SimpleStringModel(self._settings.get_as_string(self.S_URL)   or "")
        self._api_key_model    = ui.SimpleStringModel(self._settings.get_as_string(self.S_KEY)   or "")
        self._model_name_model = ui.SimpleStringModel(self._settings.get_as_string(self.S_MODEL) or "")
        # a little more complex for combo box as it needs index
        self._mode_items = ["Chat Completions", "Responses API"]
        saved = self._settings.get_as_string(self.S_MODE) or "Chat Completions"
        default_idx = self._mode_items.index(saved)
        self._mode_model = simple_combo(self._mode_items, default_idx)


        # Windows/handles
        self.window = None
        self.settings_window = None

        # Import relevant classes from other files
        self.camera = CameraControl()
        self.analyzer = ViewportAnalyzer()
        self.capturer = ViewportCapturer()

        # This is used to store last result for metrics
        self.last_result = None
        self.last_input = None

        # Initialize values in UI
        self.ai_question = ui.SimpleStringModel("")
        self.response_correctness = ui.SimpleStringModel("")
        self.navigation_correctness = ui.SimpleStringModel("")
        # Needed to stop and avoid parallel execution
        self._task: asyncio.Task | None = None


        # Main Window -------------------------------------------------------------
        self.window = ui.Window(
            "Camera Control",
            width=360,
            height=220,
            flags=ui.WINDOW_FLAGS_NO_RESIZE,
            dock_preference=ui.DockPreference.RIGHT_BOTTOM,
        )

        header_style = {"Label": {"font_size": 17, "margin_height": 2}}


        # Settings window builder ---------------------------------------------------
        def _open_settings_window():
            # If already built once, just show
            if self.settings_window is not None:
                self.settings_window.visible = True
                try:
                    self.settings_window.focus()
                except Exception:
                    pass
                return

            # Build the Settings window once
            self.settings_window = ui.Window("Settings", width=420, height=280)

            def _on_save_clicked():
                # Read UI fields
                url   = self._url_model.as_string
                key   = self._api_key_model.as_string
                model = self._model_name_model.as_string

                index = self._mode_model.index.as_int # the combo box stores the index of the selected item
                mode  = self._mode_items[index] # looking up value from index

                # Persist to carb settings
                self._settings.set(self.S_URL, url)
                self._settings.set(self.S_KEY, key)
                self._settings.set(self.S_MODEL, model)
                self._settings.set(self.S_MODE, mode)

                print(f"URL = {self._settings.get_as_string(self.S_URL)}")
                print(f"API Key = hidden")
                print(f"Model = {self._settings.get_as_string(self.S_MODEL)}")
                print(f"Mode = {self._settings.get_as_string(self.S_MODE)}")

                # Make future runs use the new settings
                self.analyzer.rebuild_ai()

                # Close after save
                self.settings_window.visible = False

            with self.settings_window.frame:
                with ui.VStack(spacing=8, height=ui.Percent(100), width=ui.Percent(100)):
                    ui.Label("LLM Settings", height=20)

                    with ui.HStack():
                        ui.Label("URL", width=90)
                        ui.StringField(model=self._url_model)

                    with ui.HStack():
                        ui.Label("API Key", width=90)
                        ui.StringField(model=self._api_key_model, password_mode=True)

                    with ui.HStack():
                        ui.Label("Model", width=90)
                        ui.StringField(model=self._model_name_model)

                    with ui.HStack():
                        ui.Label("Mode", width=90)
                        ui.ComboBox(self._mode_model) # dropdown

                    ui.Spacer()
                    with ui.HStack(height=ROW_H):
                        ui.Spacer()
                        ui.Button("Save", width=100, height=FIELD_H, clicked_fn=_on_save_clicked)
                        ui.Button("Close", width=100, height=FIELD_H,
                                  clicked_fn=lambda: setattr(self.settings_window, "visible", False))
                        ui.Spacer()

        def _click_settings_icon(*_a, **_kw):
            _open_settings_window()

        # -------------------------------------------------------------------

        # Formatting the result returned by analyze_control depending on the status
        def format_result(result: dict) -> str:
            status = result.get("status")

            if status == "error":
                return (
                    "Screenshot failed\n"
                    f"Reason: {result.get('reason', 'unknown')}"
                )

            elif status == "done":
                return (
                    "Analysis completed\n\n"
                    f"Answer:\n{result.get('answer', '')}\n\n"
                    f"Explanation:\n{result.get('final_explanation', '')}\n\n"
                )

            elif status == "max_iters_reached":
                return (
                    "Maximum iterations reached\n\n"
                    f"Last explanation:\n{result.get('last_explanation', '')}"
                )

            return "Unknown result"


        # Analyze the user question and return the final result
        async def run_analyze_control():

            """Run the analyzer loop once and display the final result.

            Args:
                None. Pulls the question from the UI.

            Returns:
                None. Displays the final result in the UI.
            """

            # Pull the question from the UI
            question = self.ai_question.get_value_as_string()

            if not question:
                self.ai_response_label.text = "Please enter a question."
                return
            self.ai_response_label.text = "Analyzing..."
            # result is a dictionary determined by analyze_control
            result = await self.analyzer.analyze_control(question)

            # Store last result and question for metrics
            self.last_result = result
            self.last_input = {"question": question}

            # Formatting dictionary returned by analyze_control based on "status"
            formatted = format_result(result)

            self.ai_response_label.text = formatted

            # Clean question afterwards
            self.ai_question.set_value("")


        def click_analyze_control():

            """
            Schedule the async analyze task without blocking the UI thread.
            """

            # # schedule the coroutine; do NOT make this function async
            # asyncio.ensure_future(run_analyze_control())

            # It doesn't let you run multiple tasks at the same time (displays text and returns early)
            if self._task and not self._task.done():
                self.ai_response_label.text = "Already running. Stop first."
                return

            async def _worker():
                try:
                    await run_analyze_control()
                # Does not await if click_stop and _stop has raised the CancelledError
                except asyncio.CancelledError:
                    # Clear UI when stopped
                    self.ai_response_label.text = ""
                    raise
                finally:
                    # Drop the handle when finished (or stopped)
                    self._task = None

            self._task = asyncio.ensure_future(_worker())


        # stop control
        async def _stop():
            if self._task and not self._task.done():
                try:
                    self._task.cancel()
                    await self._task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._task = None
            # Ensure UI is empty after stopping
            self.ai_response_label.text = ""

        def click_stop():
            asyncio.ensure_future(_stop())


        # ----------------------------------------------------------------------
        # Storing metrics to CSV file

        def save_metrics_to_csv():
            if not self.last_result or not self.last_input:
                self.save_status_label.text = "Nothing to save."
                return

            # Preparing the path
            metrics_path = os.path.join(self.data_folder, "metrics.csv")

            # Collect the active VLM model name (from settings)
            model_name = self._settings.get_as_string(self.S_MODEL) or "N/A"

            # Collecting the question
            question = self.last_input.get("question", "N/A")

            # Collectiing the correctness values
            response_corr = self.response_correctness.as_string.strip()
            navigation_corr  = self.navigation_correctness.as_string.strip()

            # Collecting the other evaluation metrics
            initial_cam_parameters = self.last_result.get("initial_cam_parameters", "{}")
            iterations = self.last_result.get("iterations", 0)
            total_translation = round(self.last_result.get("total_translation", 0), 2)
            total_rotation = round(self.last_result.get("total_rotation", 0), 2)
            elapsed_time = round(self.last_result.get("elapsed_time", 0), 2)

            # Formatting values for European deciaml comma format (needs semicolon then in CSV)
            total_translation_comma = f"{total_translation:.2f}".replace(".", ",")
            total_rotation_comma = f"{total_rotation:.2f}".replace(".", ",")
            elapsed_time_comma = f"{elapsed_time:.2f}".replace(".", ",")

            # Writing CSV (create with header if missing)
            file_exists = os.path.isfile(metrics_path)
            with open(metrics_path, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")
                if not file_exists:
                    writer.writerow([
                        "model_name",
                        "initial_cam_parameters",
                        "question",
                        "response_correctness",
                        "navigation_correctness",
                        "iterations",
                        "total_translation",
                        "total_rotation",
                        "elapsed_time",
                    ])

                writer.writerow([
                    model_name,
                    initial_cam_parameters,
                    question,
                    response_corr,
                    navigation_corr,
                    iterations,
                    total_translation_comma,
                    total_rotation_comma,
                    elapsed_time_comma,
                ])

            # Clear inputs
            self.response_correctness.set_value("")
            self.navigation_correctness.set_value("")

            # Status label + auto-clear
            self.save_status_label.text = f"Saved metrics to csv"
            # Clear after 2 seconds
            async def _clear_label():
                await asyncio.sleep(2)
                self.save_status_label.text = ""
            asyncio.ensure_future(_clear_label())



        # ----------------------------------------------------------------------
        # This allows to reset the camera to initial position (calling function with values from metrics)

        def click_reset_camera():
            self.camera.reset_initial_camera_position(
                self.last_result.get("initial_cam_parameters")
            )
            print("Camera reset to initial position.")



        # UI Layout ------------------------------------------------------------


        header_style = {
            "Label": {
                "font_size": 17,
                "margin_height": 2,
            }
        }

        with self.window.frame:
            with ui.VStack(spacing=6):

                # Settings icon and text on top -----------------------------------------
                with ui.HStack(height=20):
                    ui.Label("Ask the Vision Intelligent Assistant:", style=header_style)
                    ui.Spacer()  # pushes the icon to the right
                    ui.Image(
                        self._settings_icon_path,
                        width=18,
                        height=18,
                        mouse_pressed_fn=_click_settings_icon,
                        tooltip="Settings"
                    )
                # --------------------------------------------------------------

                with ui.Frame(height=50):
                    ui.StringField(model=self.ai_question, multiline=True)


                with ui.HStack(height=ROW_H):
                    ui.Spacer()
                    ui.Button("Apply IA", width=100, height=FIELD_H, clicked_fn=click_analyze_control)
                    ui.Button("Stop", width=80, height=FIELD_H, clicked_fn=click_stop)
                    ui.Button("Reset Camera", width=80, height=FIELD_H, clicked_fn=click_reset_camera)
                    ui.Spacer()

                ui.Label("IA Response:", height=20, style=header_style)
                with ui.ScrollingFrame(height=140):
                    self.ai_response_label = ui.Label("", word_wrap=True, alignment=ui.Alignment.LEFT_TOP)


                with ui.HStack(height=ROW_H):
                    ui.Label("Response Correctness", width=150)
                    ui.StringField(model=self.response_correctness)

                with ui.HStack(height=ROW_H):
                    ui.Label("Navigation Correctness", width=150)
                    ui.StringField(model=self.navigation_correctness)

                ui.Spacer(height=10)
                ui.Button("Save Metrics", height=28, width=140, clicked_fn=save_metrics_to_csv)
                self.save_status_label = ui.Label("", height=20)
                ui.Spacer()

    def on_shutdown(self):

        """
        Cleanup on shutdown.
        """

        try:
            if self._task and not self._task.done():
                self._task.cancel()
        except Exception:
            pass

        self.settings_window = None
        self.window = None
        self.camera = None
        self.analyzer = None
