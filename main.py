import sys
from PIL import Image
import numpy as np

# Constants
EYE_POSITION = np.array([0, 0, 0])
FORWARD_DIRECTION = np.array([0, 0, -1])
UP_DIRECTION = np.array([0, 1, 0])
RIGHT_DIRECTION = np.array([1, 0, 0])
MAX_DISTANCE = 1e8


def parse_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    spheres = []
    lights = []
    current_color = np.array([1.0, 1.0, 1.0, 1.0])
    exposure_value = None

    for line in lines:
        parts = line.strip().split()

        if not parts:
            continue

        if parts[0] == 'png':
            width = int(parts[1])
            height = int(parts[2])
            output_filename = parts[3]

        elif parts[0] == 'sphere':
            sphere = {
                'center': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                'radius': float(parts[4]),
                'color': current_color
            }
            spheres.append(sphere)

        elif parts[0] == 'color':
            current_color = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3]), 1.0])

        elif parts[0] == 'sun':
            light_direction = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])])
            # Negative because it's light coming in
            lights.append(
                {'direction': -light_direction, 'color': current_color})

        elif parts[0] == 'expose':
            exposure_value = float(parts[1])

        else:
            print(f"Unknown command {parts[0]}")

    return width, height, output_filename, spheres, lights, exposure_value


def apply_exposure(color, exposure_value):
    """
    Applies exposure to a linear color value.

    Args:
        color (numpy.ndarray): Linear color value (RGB)
        exposure_value (float): Exposure value. If None, no exposure adjustment is applied.

    Returns:
        numpy.ndarray: Exposed color value
    """
    if exposure_value is not None:
        return 1 - np.exp(-exposure_value * color)
    return color


def normalize(vector):
    return vector / np.linalg.norm(vector)


def gamma_encode(linear_value):
    """
    Performs RGB to sRGB conversion for linear-space illumination 

    Args:
        linear_value (numpy.float64): RGB value

    Returns:
        numpy.float64: sRGB value
    """

    if linear_value <= 0.0031308:
        return 12.92 * linear_value
    else:
        return 1.055 * (linear_value ** (1.0 / 2.4)) - 0.055


def intersect_ray_sphere(ray_origin, ray_dir, sphere):
    """
    Performs Ray-Sphere Intersection Algorihtm

    Args:
        ray_origin (numpy.ndarray): origin of the ray
        ray_dir (numpy.ndarray): unit-length direciton of the ray
        sphere (dict): Sphere with center $c$ and radius $r$. 

    Returns:
        numpy.float64: intersection distance $t$
        numpy.ndarray: intersection point $p$
    """

    r = sphere['radius']
    c = sphere['center']

    # Find the length between the sphere center and the ray origin
    L = c - ray_origin

    # Find t_c, the t value of the point
    # where the ray comes closest to the center of the sphere
    tc = np.dot(L, ray_dir)

    # Define is_inside, the distance along the ray to $\mathbf{c'}$,
    # the rays’ closest approach to $\mathbf c$,
    # but we don’t need $\mathbf{c'}$, only $t_c$
    is_inside = np.dot(L, L) < (r * r)

    # if not inside and  tc < 0, there's no intersection
    if not is_inside and tc < 0:
        return MAX_DISTANCE, None

    # if not inside and  r**2 < d**2, there's no intersection
    d_square = np.dot(L, L) - (tc * tc)
    if not is_inside and d_square > (r * r):
        return MAX_DISTANCE, None

    # Define t_offset, the difference between $t$ and $t_c$;
    # rays intersect spheres twice (once entering, once exiting)
    # so $t_c \pm t_{\text{offset}}$ are both intersection points
    t_offset = np.sqrt((r * r) - d_square)

    if is_inside:
        t = tc + t_offset
    else:
        t = tc - t_offset

    intersection_point = ray_origin + ray_dir * t

    return t, intersection_point


def ray_collision(ray_origin, ray_dir, scene_objects, ignore_object=None, max_distance=MAX_DISTANCE):
    closest_t = max_distance
    closest_object = None
    for obj in scene_objects:
        if obj is ignore_object:
            continue  # Ignore self in shadow checks
        t, _ = intersect_ray_sphere(ray_origin, ray_dir, obj)
        if 0 < t < closest_t:
            closest_t = t
            closest_object = obj
    return closest_object, closest_t


def render(width, height, spheres, lights, exposure_value):
    # Initialize image
    image = np.zeros((height, width, 4))  # Includes alpha channel

    for y in range(height):
        for x in range(width):
            # Compute ray direction for pixel
            sx = (2.0 * x - width) / max(width, height)
            sy = (height - 2.0 * y) / max(width, height)

            ray_dir = normalize(FORWARD_DIRECTION +
                                sx * RIGHT_DIRECTION +
                                sy * UP_DIRECTION)

            # Find the closest object the ray intersects
            closest_sphere, closest_t = ray_collision(
                EYE_POSITION, ray_dir, spheres)

            if closest_sphere is not None:
                intersection_point = EYE_POSITION + closest_t * ray_dir
                normal = normalize(intersection_point -
                                   closest_sphere['center'])

                # Use the sphere's RGBA color but ignore the alpha while computing lighting
                base_color = closest_sphere['color'][:3]
                object_alpha = closest_sphere['color'][3]

                color = np.zeros(3)  # Initialize color for RGB components

                for light in lights:
                    # Light direction is from light to point
                    light_dir = -normalize(light['direction'])
                    shadow_offset = intersection_point + 1e-5 * \
                        normal  # Offset to prevent self-shadowing
                    blocking_object, shadow_t = ray_collision(
                        shadow_offset, light_dir, spheres, ignore_object=closest_sphere)

                    # If no blocking object on the way to the light, calculate the light contribution
                    if blocking_object is None:
                        lambertian = max(np.dot(light_dir, normal), 0)
                        # Work with the RGB components only
                        color += base_color * light['color'][:3] * lambertian

                # Ensure color values are within [0, 1]
                color = np.clip(color, 0, 1)

                # Apply exposure
                color = apply_exposure(color, exposure_value)

                # Apply gamma correction to RGB
                image[y, x, :3] = [gamma_encode(c) for c in color]

                # Set alpha channel based on object's alpha
                image[y, x, 3] = object_alpha

            else:
                # Set the pixel to black and fully transparent
                # Background is fully transparent
                image[y, x] = np.array([0, 0, 0, 0])

    return (image * 255).astype(np.uint8)


def main():
    if len(sys.argv) != 2:
        print("Usage: ./yourprogram inputfile.txt")
        sys.exit(1)

    input_file_path = sys.argv[1]
    width, height, output_filename, spheres, lights, exposure_value = \
        parse_input(input_file_path)

    image_data = render(width, height, spheres, lights, exposure_value)
    image = Image.fromarray(image_data, 'RGBA')
    image.save(output_filename)


if __name__ == '__main__':
    main()
