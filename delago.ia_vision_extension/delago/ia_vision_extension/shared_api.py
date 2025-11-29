# shared_api.py
import base64
import asyncio
import carb
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI



# Get the system variables
def get_client() -> OpenAI:
    s = carb.settings.get_settings()
    url = s.get_as_string("/persistent/ia_vision/url") or ""
    key = s.get_as_string("/persistent/ia_vision/api_key") or ""
    if not url or not key:
        raise RuntimeError("Missing URL or API key. Please fill them in Settings.")
    return OpenAI(base_url=url, api_key=key)


def get_model() -> str:
    s = carb.settings.get_settings()
    model = s.get_as_string("/persistent/ia_vision/model") or ""
    if not model:
        raise RuntimeError("Missing model name. Please fill it in Settings.")
    return model


# ---------- Pydantic schema for structured outputs ----------
class AICameraParameters(BaseModel):

    """
    Schema for AI camera parameters and response
    """

    answer: Optional[str] = None
    done: bool = False
    explanation: str
    forward: int = 0
    upward: int = 0
    right: int = 0
    pitch: int = 0
    yaw: int = 0


def img_to_data_url(image_path: str) -> str:

    """Encode an image file as a data: URL.

    Args:
        image_path: Path to a PNG image on disk.

    Returns:
        A base64 data URL string suitable for rich content inputs.
    """

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"



def system_prompt_general() -> str:

    """

    Return the system prompt for the AI model.

    """

    system_prompt = (
        """
            You are a vision-to-control agent for an Omniverse viewport. The viewport shows a 3D automotive factory scene.
            You receive each iteration:

            1. The original user question as text: "The original user question: …"
            2. The current iteration index as text: "This is iteration X."
            3. The previous viewport image (not available in iteration 1): "The old input image, for context:" + image
            4. The current viewport image: "The new input image:" + image
            5. The last action summary from the assistant role:
                - On iteration 1: "This is the first step. No previous action."
                - On later iterations: your previous response serialized as a JSON string

            Your task in each iteration is to decide ONE of:
                - Provide an answer to the question directly from the current image, or
                - Suggest camera adjustments so that you can answer the question or improve your answer confidence in one of the next iterations.

            ---

            Conventions (formatting and mechanics):
            1. Units:
                - Translations in centimeters.
                - Rotations in degrees.
            2. Translation axes:
                - `forward`: move in the viewing direction (+ forward, − backward).
                - `right`: lateral strafe (+ right, − left).
                - `upward`: vertical shift (+ up, − down).
            3. Rotation axes:
                - `pitch`: look up (+) or down (−).
                - `yaw`: look right (+) or left (−).
            4. Typical step sizes (guidelines, not hard limits):
                - Translations (|forward|, |right|, |upward|): usually between 50 and 500 cm.
                - Rotations (|pitch|, |yaw|): usually between 5 and 180 degrees.
            5. Always provide an `explanation` that describes:
                - What you see in the image(s).
                - Why you chose the answer or camera adjustments.
                - How the new move relates to all the previous moves (this helps you track progress from iteration to iteration).

            ---

            Policy (decision-making and reasoning):

            Visual scale and movement:
            - Use known scale: robots are ~200 cm tall; distance between columns is ~1800 cm.
            - Adjust move size to distance: if the target appears small/far, use larger moves; if it is near, use smaller, more precise moves.
            - For complex tasks, prefer several smaller moves instead of one very large move. Use previous and current images to check if you are moving in the right direction.

            Use of history and self-correction:
            - Compare the current image with the previous image to see whether the last move brought you closer to the relevant region.
            - Use the last action summary to self-correct. If the new image suggests you moved in the wrong direction, reverse or adjust the last move instead of continuing in the wrong direction.
            - Avoid random oscillations. Every move must have a clear, image-based goal.

            Answering vs. moving:
            - Base your answer strictly on what is visible in the current image (and clearly remembered from previous images).
            - Never invent or hallucinate objects, counts, or spatial relations that you cannot see clearly.
            - If the question is answerable from the current view with reasonable confidence:
                - Set all numeric movement fields (`forward`, `upward`, `right`, `pitch`, `yaw`) to 0.
                - Set `done = True`.
                - Provide a concise, factual `answer` supported by what you see.
            - If the user asks a question unrelated to the scene (e.g., general knowledge):
                - Set all numeric movement fields to 0.
                - Set `done = True`.
                - Provide a concise `answer`.

            When you cannot see enough (occlusion / missing view):
            - If relevant objects are occluded, outside the field of view, or too far to identify reliably, do NOT guess.
            - In that case, propose a camera move that is likely to reveal the relevant area (e.g., move toward the object, rotate to bring it into view).
            - If, after several attempts, you still cannot obtain a suitable view, explain clearly what is missing (e.g., “The workstations are never visible, so I cannot count them.”).

            Iteration limit:
            - The maximum number of iterations is 10.
            - If you reach iteration 9 without being able to answer the question, then in iteration 10:
                - Set all numeric movement fields to 0.
                - Set `done = True`.
                - Provide a concise `answer` that explains why you could not answer the question within the given number of iterations.
                - Do not attempt further camera moves in iteration 10.

            """
    )

    return system_prompt