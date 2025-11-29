# camera_control.py

import omni.usd
from pxr import UsdGeom, Gf, Usd
from omni.kit.viewport.utility import get_active_viewport_camera_path
import json

class CameraControl:

    """
    Applies AI-suggested translate/rotate deltas to the active camera.
    """

    # Apply AI-suggested deltas to the active camera (in current viewing direction)
    def apply_translate_and_rotate_ai(self, camera_params):

        """Apply one translate/rotate step to the active viewport camera.

        Args:
            class AICameraParameters(BaseModel):
            answer: Optional[str] = None
            done: bool = False
            explanation: str
            forward: int = 0
            upward: int = 0
            right: int = 0
            pitch: int = 0
            yaw: int = 0

        Returns:
            None. Applies changes directly to the active camera.
        """

        # This is the x axis in USD camera space (cm). LLM returns +right, -left as more intuitive
        move_left = -camera_params.right
        # This is the y axis in USD camera space (cm)
        move_up = camera_params.upward
        # This is the z axis in USD camera space (cm)
        move_forward = camera_params.forward

        # Rotation deltas (degrees)
        watch_up = camera_params.pitch # pitch
        watch_left = -camera_params.yaw # yaw. LLM returns +watch right, - watch left as more intuitive

        # Create a local vector in camera space [x,y,-z] https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html
        local_vector = Gf.Vec3d(-move_left, move_up, -move_forward)

        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        cam_path = get_active_viewport_camera_path()
        cam_prim = stage.GetPrimAtPath(str(cam_path))

        old_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        try:
            cam_xf = UsdGeom.Xformable(cam_prim)

            # Translation: read, add deltas, write; do NOT touch rotation ops
            if abs(move_left) > 0 or abs(move_up) > 0 or abs(move_forward) > 0:

                t_attr = cam_prim.GetAttribute("xformOp:translate")
                if not (t_attr and t_attr.IsValid()):
                    cam_xf.AddTranslateOp()
                    t_attr = cam_prim.GetAttribute("xformOp:translate")

                current_t = t_attr.Get()
                if current_t is None:
                    current_t = Gf.Vec3d(0, 0, 0)

                # Matrix to transform from local camera space to world space
                local_to_world_matrix = cam_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

                # Matrix multiplication y = Mx   with x as the vector in local camera space
                world_x = local_to_world_matrix[0][0]*local_vector[0] + local_to_world_matrix[1][0]*local_vector[1] + local_to_world_matrix[2][0]*local_vector[2]
                world_y = local_to_world_matrix[0][1]*local_vector[0] + local_to_world_matrix[1][1]*local_vector[1] + local_to_world_matrix[2][1]*local_vector[2]
                world_z = local_to_world_matrix[0][2]*local_vector[0] + local_to_world_matrix[1][2]*local_vector[1] + local_to_world_matrix[2][2]*local_vector[2]


                new_t = Gf.Vec3d(
                    int(current_t[0]) + int(world_x),
                    int(current_t[1]) + int(world_y),
                    int(current_t[2]) + int(world_z)
                )


                t_attr.Set(new_t)


            # Rotation: read, add deltas, write; do NOT touch translation ops
            if abs(watch_up) > 0 or abs(watch_left) > 0:
                # Prefer existing rotateXYZ; avoid clobbering quaternion orient if that exists also.
                r_attr = cam_prim.GetAttribute("xformOp:rotateXYZ")
                if r_attr and r_attr.IsValid() and r_attr.Get() is not None:
                    current_r = r_attr.Get()
                    if current_r is None:
                        current_r = Gf.Vec3d(0, 0, 0)

                    new_r = Gf.Vec3d(
                        int(current_r[0]) + int(watch_up),      # pitch (X)
                        int(current_r[1]),                      # keep roll (Y)
                        int(current_r[2]) + int(watch_left)     # yaw (Z)
                    )
                    r_attr.Set(new_r)


        finally:
            stage.SetEditTarget(old_target)


    # These are two helper function to make the experiments easier to log and reproduce

    def get_initial_camera_position(self) -> tuple[Gf.Vec3d, Gf.Vec3d]:
        """Get the current position and rotation of the active viewport camera.

        Returns:
            A tuple containing:
                - position (Gf.Vec3d): The current position as a Gf.Vec3d.
                - rotation (Gf.Vec3d): The current rotation as a Gf.Vec3d (rotateXYZ in degrees).
        """

        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        cam_path = get_active_viewport_camera_path()
        cam_prim = stage.GetPrimAtPath(str(cam_path))

        translation = Gf.Vec3d(cam_prim.GetAttribute("xformOp:translate").Get())
        rotation = Gf.Vec3d(cam_prim.GetAttribute("xformOp:rotateXYZ").Get())

        return translation, rotation


    def reset_initial_camera_position(self, initial_cam_parameters: str):
        """Reset the active viewport camera to a given position and rotation.

        Args:
            initial_cam_parameters (str): JSON string storing initial camera parameters in the format
                {"translation":[x,y,z],"rotation":[rx,ry,rz]}.

        Returns:
            None. Applies changes directly to the active camera.
        """

        print(initial_cam_parameters)

        cam_data = json.loads(initial_cam_parameters)

        # This rebuilds the Vec3d from the dumped json data
        t = cam_data.get("translation")
        r = cam_data.get("rotation")
        initial_translation = Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))
        initial_rotation    = Gf.Vec3d(float(r[0]), float(r[1]), float(r[2]))

        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        cam_path = get_active_viewport_camera_path()
        cam_prim = stage.GetPrimAtPath(str(cam_path))

        old_target = stage.GetEditTarget()
        stage.SetEditTarget(stage.GetSessionLayer())
        try:
            # Ensuring that the translate and rotation op exist
            cam_xf = UsdGeom.Xformable(cam_prim)

            t_attr = cam_prim.GetAttribute("xformOp:translate")
            if not (t_attr and t_attr.IsValid()):
                cam_xf.AddTranslateOp()
                t_attr = cam_prim.GetAttribute("xformOp:translate")

            r_attr = cam_prim.GetAttribute("xformOp:rotateXYZ")
            if not (r_attr and r_attr.IsValid()):
                cam_xf.AddRotateXYZOp()
                r_attr = cam_prim.GetAttribute("xformOp:rotateXYZ")

            # Applying the initial camera parameters
            t_attr.Set(initial_translation)
            r_attr.Set(initial_rotation)

            print("Camera reset to:", t_attr.Get(), r_attr.Get())
        finally:
            stage.SetEditTarget(old_target)
