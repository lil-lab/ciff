import math


def get_image_coordinates(object_distance, object_angle, object_height,
                          camera_angle, image_height, image_width):
    """ Given object distance and angles in degree! computes mapping of the object on the image as a pixel.
        Object's distance and angle are in polar coordinate with object angle 0 if the object is directly infront
        of the agent and measured positive clockwise.
        Returns the pixel coordinates. """

    half_image_height = image_height/2.0
    half_image_width = image_width/2.0
    half_camera_angle = camera_angle/2.0
    # object_height = - camera_height

    if object_angle < - half_camera_angle or object_angle > half_camera_angle:
        return None, None

    tan_theta = math.tan(math.radians(half_camera_angle))
    cos_phi = math.cos(math.radians(object_angle))
    tan_phi = math.tan(math.radians(object_angle))

    try:
        row_val = half_image_height - (half_image_height * object_height)/(object_distance * cos_phi * tan_theta)
        col_val = half_image_width + (half_image_width * tan_phi) / tan_theta
    except Exception:
        return None, None

    row = int(row_val)
    col = int(col_val)

    if row_val < 0 or row_val >= image_height:
        return None, None
    if col_val < 0 or col_val >= image_width:
        return None, None

    return row, col


def get_inverse_object_position(row_mean, col_mean, object_height,
                                camera_angle, image_height, image_width, agent_location):
    """ Given object's pixel position, find its location in the real world.
    Assumes that the object height is known """

    half_image_height = image_height / 2.0
    half_image_width = image_width / 2.0
    tan_theta = math.tan(math.radians(camera_angle))

    s = half_image_height / (row_mean - half_image_height)
    delta_z_local = (object_height/tan_theta) * s
    delta_x_local = object_height * (col_mean - half_image_width) / (row_mean - half_image_height) *\
                    (image_height/image_width)

    x, z, angle = agent_location

    sin_pose = math.sin(math.radians(angle))
    cos_pose = math.cos(math.radians(angle))

    delta_x_global = delta_z_local * sin_pose + delta_x_local * cos_pose
    delta_z_global = delta_z_local * cos_pose - delta_x_local * sin_pose

    x_pred_global = x + delta_x_global
    z_pred_global = z + delta_z_global

    return x_pred_global, z_pred_global
