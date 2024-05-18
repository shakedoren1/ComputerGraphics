from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # reflection variable initialized
            reflection = 1.0

            # This is the main loop where each pixel color is computed.
            for _ in range(max_depth):
                # Find the closest intersection with an object
                nearest_object, _, intersection = ray.nearest_intersected_object(objects)
                # if there is no intersection, continue to next pixel
                if nearest_object is None:
                    break
                # add a small shift to the intersection point to avoid self-intersection
                normal = nearest_object.get_normal(intersection)
                intersection += 1e-4 * normal                

                # Find the color of the pixel
                illumination = get_color(origin, ambient, lights, objects, nearest_object, normal, intersection)

                # reflection calculation
                color += reflection * illumination
                reflection *= nearest_object.reflection
                ray = Ray(intersection, reflected(ray.direction, normal))
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image

# A function that calculates the color of a pixel according to the Phong reflection model
def get_color(origin, ambient, lights, objects, nearest_object, normal, intersection):
    color = np.zeros(3, dtype=np.float64)
    
    # Ambient calculation
    color = calc_ambient_color(nearest_object, ambient).astype(np.float64)

    for light in lights:
        light_ray = light.get_light_ray(intersection)
        # Check if the point is shadowed
        if is_shadowed(light, light_ray, intersection, objects):
            continue
        # Diffuse calculation
        diffuse_light = calc_diffuse_color(nearest_object, normal, light_ray).astype(np.float64)
        # Specular calculation
        specular_light = calc_specular_color(nearest_object, normal, light_ray, origin, intersection).astype(np.float64)
        # Add the light to the color
        color += (diffuse_light + specular_light).astype(np.float64) * light.get_intensity(intersection).astype(np.float64)

    # The color received
    return color

# A function that calculates the ambient color of an object
def calc_ambient_color(nearest_object, ambient):
    return nearest_object.ambient * ambient

# A function that checks if a point is shadowed by an array of objects
def is_shadowed(light, light_ray, intersection, objects):
    result = light_ray.nearest_intersected_object(objects)
    if result is None:
        return False
    nearest_object, t, _ = result
    distance_from_light = light.get_distance_from_light(intersection)
    if nearest_object is not None and t < distance_from_light:
        return True
    return False

# A function that calculates the diffuse color of an object
def calc_diffuse_color(nearest_object, normal, light_ray):
    return np.array(nearest_object.diffuse) * (np.dot(light_ray.direction, normal))

# A function that calculates the specular color of an object
def calc_specular_color(nearest_object, normal, light_ray, origin, intersection):
    V = normalize(origin - intersection)
    R = normalize(reflected(light_ray.direction, normal))
    return np.array(nearest_object.specular) * ((np.dot(V, R)) ** nearest_object.shininess)

# Write your own objects and lights
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    # Objects
    # Background Plane
    background = Plane([0, 0, 1], [0, 0, -2])  # Normal points towards the camera, positioned slightly away
    background.set_material([0.3, 0.6, 0.9], [0.3, 0.6, 0.9], [0.1, 0.1, 0.9], 10, 0)  # Light blue color for the background
    objects.append(background)

    # Left eye
    left_eye = Sphere([-0.4, 0.6, -0.5], 0.2)
    left_eye.set_material([0.5, 1, 0.5], [0.5, 1, 0.5], [0.5, 1, 0.5], 5, 0.1)
    objects.append(left_eye)
    # Right eye
    right_eye = Sphere([0.4, 0.6, -0.5], 0.2)
    right_eye.set_material([0.5, 1, 0.5], [0.5, 1, 0.5], [0.5, 1, 0.5], 5, 0.1)
    objects.append(right_eye)
    # Mouth
    mouth = Triangle([-0.2, 0.1, -0.5], [0.2, 0.1, -0.5], [0, -0.1, -0.5])
    mouth.set_material([0.5, 1, 0.5], [0.5, 1, 0.5], [0.5, 1, 0.5], 5, 0.1)
    objects.append(mouth)

    # Ground Plane (for reflection)
    ground = Plane([0, 2, 0.3], [0, -1, 0])  # Normal points upwards, positioned below the entire scene
    ground.set_material([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 10, 0.5)
    objects.append(ground)

    # Lights
    # Spot Light in the middle of the scene
    spotlight = SpotLight(intensity= np.array([0.6, 0.6, 0.6]),position=np.array([0,0,0]), direction=([0,0,-1]), kc=0.1,kl=0.1,kq=0.1)
    lights.append(spotlight)

    return camera, lights, objects



