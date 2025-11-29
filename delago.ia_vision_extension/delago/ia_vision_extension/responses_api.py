# ai_control.py
import base64
import asyncio
from typing import Optional
from pydantic import BaseModel
from .shared_api import img_to_data_url, AICameraParameters, system_prompt_general, get_client, get_model



class AIControlResponsesAPI:

    """
    Wrapper around the vision LLM call for structured outputs via Response API.
    """

    def __init__(self):

        """
        Initialize client, model and system prompt.
        """

        self.AICameraParameters = AICameraParameters
        self.client = get_client()
        self.model = get_model()
        # Specify low, medium, or high for this parameter
        self.reasoning={"effort": "low"}

        pydantic_addon ="""
            Output contract:
            - Use the provided Pydantic model `AICameraParameters` for your response.
            - Return only the model fields. Do not include extra keys or text.
            """

        self.system_prompt = system_prompt_general() + "\n" + pydantic_addon


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
        input_history = [{"type":"input_text","text": "The original user question:"},
                         {"type":"input_text","text": question},
                         {"type":"input_text","text": f"This is iteration {iteration}."}]

        if previous_image_path:
            input_history.extend([{"type":"input_text","text": "The old input image, for context:"},
                                 {"type":"input_image","image_url": img_to_data_url(previous_image_path)}])
        input_history.extend([{"type":"input_text","text": "The new input image:"},
                             {"type":"input_image","image_url": img_to_data_url(image_path)}])

        # Creating response history
        if last_action_summary:
            response_history = [{"type":"output_text","text": last_action_summary}]
        else:
            response_history = [{"type":"output_text","text": "This is the first step. No previous action."}]


        # Calling the client with sytem prompt, input history (including current and previous image), and response history:
        resp = await asyncio.to_thread(
            lambda: self.client.responses.parse(
                model=self.model,
                reasoning=self.reasoning,
                input=[
                    {"role":"system","content": self.system_prompt},
                    {"role":"user","content": input_history},
                    {"role":"assistant","content": response_history}
                ],
                text_format=self.AICameraParameters,
            )
        )
        return resp.output_parsed
