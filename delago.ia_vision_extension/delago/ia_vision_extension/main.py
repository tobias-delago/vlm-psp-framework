# main.py
from .viewport_capture import ViewportCapturer
from .responses_api import AIControlResponsesAPI
from .base_api import AIControlBaseAPI
from .camera_control import CameraControl
from omni.kit.app import get_app
import omni.usd
from omni.kit.viewport.utility import get_active_viewport_camera_path
import time
import json
import asyncio
import carb

# Conceiving the vision intellignet assistant loop -----------------------------------------------------

# 1. Loop is triggered by button in ui which also pulls the user question
# 2. Tool "viewport_capture" is called, returns image path
# 3. Sends image to LLM "ai_control" with user question, and if available previous context (previous images, previous responses)
# 4. LLM deciced if it can answer question with current view or if it needs to move
# 5. If sufficiently good answer possible, leave camera parameters at 0, return answer, and set flag to "done"
# 6. If camera move needed, return camera deltas and explanation
# 7. Tool "camera_control" is called with delta, moves camera
# 8. Loop back to step 2, until "done" flag is set or max iterations reached
# Author's note: it is important that LLM calls down the loop receive all the information from previous steps (original question, previous images, previous responses)
# Author's note: ideally the number of iterations is also provided to the LLM, so it sees if it is running in circles


# decide which AIControl to use based on mode settings (currently either "Responses API" or "Chat Completions")
def get_mode() -> str:
    s = carb.settings.get_settings()
    mode = s.get_as_string("/persistent/ia_vision/mode")

    match mode:
        case "Responses API":
            return AIControlResponsesAPI()
        case "Chat Completions":
            return AIControlBaseAPI()
        case _:
            # error if a new mode name is saved but not implemented
            raise ValueError(f"Unknown mode in settings: {mode}")


# main.py
class ViewportAnalyzer:

    """
    Coordinator for capture → LLM → camera control → loop.
    """

    def __init__(self):

        """
        Instantiate capturer, LLM control, and camera control helpers.
        """

        self.capturer = ViewportCapturer()
        self.ai = get_mode()
        self.camera = CameraControl()

    def rebuild_ai(self):
        """Recreate AIControl so new settings take effect next run."""
        self.ai = get_mode()


    async def analyze_control(self, question: str):

        """Iterate capture and control until the LLM can answer or limit is hit.

        Args:
            question: User question about the current scene.

        Returns:
            Dict with one of:
                - {"status":"done","answer":str|None,"final_explanation":str,"initial_cam_parameters":str,"iterations":int,"total_translation":float,"total_rotation":float,"elapsed_time":float}
                - {"status":"error","reason":"screenshot_failed","initial_cam_parameters":str,"iterations":int,"total_translation":float,"total_rotation":float,"elapsed_time":float}
                - {"status":"max_iters_reached","last_explanation":str,"initial_cam_parameters":str,"iterations":int,"total_translation":float,"total_rotation":float,"elapsed_time":float}

            Also writes debug info to console at each iteration.
        """

        # Loop parameters
        max_iters = 10
        prev_img = None
        last_action_summary = None

        # Intialize also the evaluation metrics before new loop
        start_time = time.perf_counter()
        total_translation = 0.0
        total_rotation = 0.0

        # Get intial transformation and rotation for evaluation metrics
        translation, rotation = self.camera.get_initial_camera_position() # here we access the created camera control class instance
        initial_cam_parameters = json.dumps({
            "translation": list(translation),
            "rotation": list(rotation)
        })

        async def _wait_frames(n_frames: int):
            """
            Wait for n frames to allow the viewport to update after camera move.
            """
            app = get_app()
            for _ in range(n_frames):
                await app.next_update_async()
                #await asyncio.sleep(0.5)

        # Main loop
        for i in range(1, max_iters + 1):
            img_path = await self.capturer.save_async()
            if not img_path:
                elapsed_time = time.perf_counter() - start_time
                return {
                    "status": "error",
                    "reason": "screenshot_failed",
                    "initial_cam_parameters": initial_cam_parameters,
                    "iterations": i,
                    "total_translation": total_translation,
                    "total_rotation": total_rotation,
                    "elapsed_time": elapsed_time
                    }

            # Calls the ai and returns the Pydantic model
            ai_response = await self.ai.call_vision_llm_async(
                image_path=img_path,
                question=question,
                iteration = i,
                previous_image_path=prev_img,
                last_action_summary=last_action_summary

            )

            print(f"AI response at iteration {i}: {ai_response.model_dump_json()}")

            # if the ai has set the "done" flag to True, return the answer
            if ai_response.done:
                elapsed_time = time.perf_counter() - start_time
                return {
                    "status": "done",
                    "answer": ai_response.answer,
                    "final_explanation": ai_response.explanation,
                    "initial_cam_parameters": initial_cam_parameters,
                    "iterations": i,
                    "total_translation": total_translation,
                    "total_rotation": total_rotation,
                    "elapsed_time": elapsed_time
                }

            # Else, apply the camera move and continue the loop with updated history (do not return anything yet)
            else:
                # first update the evaluation metrics (translation as euclidean distance in xyz, rotation as sum of absolute pitch and yaw)
                step_translation = (ai_response.forward ** 2 + ai_response.upward ** 2 + ai_response.right ** 2) ** 0.5
                total_translation += step_translation
                step_rotation = abs(ai_response.pitch) + abs(ai_response.yaw)
                total_rotation += step_rotation

                # then apply the camera move
                self.camera.apply_translate_and_rotate_ai(ai_response)
                await _wait_frames(12) # waiting time depends on frame rate (e.g. 12 frames = 0.4s at 30fps)
                prev_img = img_path
                last_action_summary = ai_response.model_dump_json()



        # if done flag was not set after max iters, return status
        elapsed_time = time.perf_counter() - start_time
        return {
            "status": "max_iters_reached",
            "last_explanation": ai_response.explanation,
            "initial_cam_parameters": initial_cam_parameters,
            "iterations": i,
            "total_translation": total_translation,
            "total_rotation": total_rotation,
            "elapsed_time": elapsed_time
        }
