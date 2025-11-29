# ai_control.py
import base64
import asyncio
from pydantic import BaseModel, ValidationError
import json
from .shared_api import img_to_data_url, AICameraParameters, system_prompt_general, get_client, get_model



class AIControlBaseAPI:

    """
    Wrapper around the vision LLM call for structured outputs via Base API.
    """

    def __init__(self):

        """
        Initialize client, model and system prompt.
        """

        # Also needed in base API because of model output validation
        self.AICameraParameters = AICameraParameters
        self.client = get_client()
        self.model = get_model()

        json_addon = """
            Output contract (STRICT):
            - Respond ONLY with a single JSON object (no markdown fences, no prose).
            - Keys and types must be EXACTLY:

            {
            "answer": string or null,
            "done": boolean,
            "explanation": string,
            "forward": integer,
            "upward": integer,
            "right": integer,
            "pitch": integer,
            "yaw": integer
            }

            Rules:
            - Do not include any extra keys.
            - All numeric fields must be integers (no floats, no strings).
            - If you do not need to move the camera, set all numeric fields to 0.
            - If you can answer now, set "done": true and include "answer".
            - If you cannot answer now, set "done": false and set movement fields accordingly.
            - Output must be valid JSON on a single line or compact pretty-JSON; no comments, no explanations.
            """
        self.system_prompt = system_prompt_general() + "\n" + json_addon


    async def call_vision_llm_async(
            self,
            image_path: str,
            question: str,
            iteration: int,
            previous_image_path: str | None = None,
            last_action_summary: str | None = None) -> AICameraParameters:

        """Call the structured vision model with images and history.

        Args:
            image_path: Path to the current viewport screenshot.
            question: Original user question about the scene.
            iteration: 1-based iteration counter for the loop.
            previous_image_path: Optional path to the prior screenshot.
            last_action_summary: Optional serialized summary of the last step.

        Returns:
            AICameraParameters parsed from the model response.
        """

        # Creating input history
        input_history = [{"type":"text","text": "The original user question:"},
                         {"type":"text","text": question},
                         {"type":"text","text": f"This is iteration {iteration}."}]

        if previous_image_path:
            input_history.extend([{"type":"text","text": "The old input image, for context:"},
                                 {"type": "image_url", "image_url": {"url": img_to_data_url(previous_image_path)}}])
        input_history.extend([{"type":"text","text": "The new input image:"},
                             {"type": "image_url", "image_url": {"url": img_to_data_url(image_path)}}])

        # Creating response history
        if last_action_summary:
            response_history = [{"type":"text","text": last_action_summary}]
        else:
            response_history = [{"type":"text","text": "This is the first step. No previous action."}]


        # Calling the client with sytem prompt, input history (including current and previous image), and response history:
        resp = await asyncio.to_thread(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role":"system","content": self.system_prompt},
                    {"role":"user","content": input_history},
                    {"role":"assistant","content": response_history}
                ],
            )
        )


        raw = resp.choices[0].message.content

        try:
            response_pydantic = AICameraParameters.model_validate_json(raw)
            return response_pydantic
        except ValidationError as e:
            print("Validation failed:", e)
            raise
