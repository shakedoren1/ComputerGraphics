from typing import List

from helper_classes import *
import matplotlib.pyplot as plt
import numpy as np


def render_scene(camera: np.array, ambient: np.array, lights: List[LightSource], objects: List[Object3D], screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    # we want the screen to have the same aspect ratio than the actual image we want to produce -> 2/ (2/ratio) = ratio
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))
    # For each pixel
    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # This is the main loop where each pixel color is computed.
            # pixel has z=0 since it lies on the screen which is contained in the plane formed by the x and y axes
            pixel = np.array([x, y, 0])
            color = np.zeros(3)
            reflection = 1

            # 1. shoot a ray from the center of projection throughout the pixel
            ray = construct_ray_throughout_pixel(camera, pixel)

            # 2. calc intersection
            for _ in range(max_depth):
                # 2. construct the closest intersection with scene object
                # check for intersections, min_distance = t -> for ray = origin + t*direction_vector
                t, nearest_object = ray.nearest_intersected_object(objects)
                if nearest_object is None:
                    break  # if nearest object is none -> we didn't found any intersections with the ray -> continue to next pixel

                # compute intersection point between ray and nearest object
                intersection_point = ray.get_intersection_point(t)
                normal_to_surface = np.array(nearest_object.get_normal_to_surface(intersection_point))
                shifted_point = intersection_point + SHIFT_CONST * normal_to_surface

                # 3. get color
                illumination = calc_color(nearest_object, shifted_point, lights, objects, camera, ambient, normal_to_surface)
                # reflection
                color += reflection * illumination
                reflection *= nearest_object.reflection

                ray = Ray(shifted_point, reflected(ray.direction, normal_to_surface))

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def construct_ray_throughout_pixel(origin: np.array, pixel: np.array) -> Ray:
    """ This function construct a ray that comes out of the camera towards the pixel"""
    direction = normalize(pixel - origin)
    return Ray(origin, direction)


def calc_color(nearest_object: Object3D, intersection_point: np.array, lights, objects, camera, ambient, normal_to_surface):
    """this function calculate the pixel color according the phong reflection model"""
    color = np.zeros((3))
    # ambiant
    color += nearest_object.ambient * ambient
    for light_source in lights:
        is_shadowed = is_occluded_by(light_source, intersection_point, objects)
        if is_shadowed:
            continue

        light_ray = light_source.get_light_ray(intersection_point)
        # diffuse
        diffuse_light = nearest_object.diffuse * (np.dot(light_ray.direction, normal_to_surface))
        # specular
        V = normalize(camera - intersection_point)
        R = normalize(reflected(light_ray.direction, normal_to_surface))
        specular_light = nearest_object.specular * ((np.dot(V, R)) ** nearest_object.shininess)
        color += (diffuse_light + specular_light) * light_source.get_intensity(intersection_point)
    return color


def is_occluded_by(light_source: LightSource, intersection_point, objects):
    """
    Checks if one of the scene objects occludes the light-source. an object occludes the light source
     if the light ray first intersects the surface before reaching the light source.
     returns 0 if one of the scene objects is shadowing the intersection point, 1 otherwise
    """
    ray_to_light = light_source.get_light_ray(intersection_point)
    min_distance_to_light, _ = ray_to_light.nearest_intersected_object(objects)
    distance_intersection_from_light = light_source.get_distance_from_light(intersection_point)
    return min_distance_to_light and min_distance_to_light < distance_intersection_from_light


# Write your own objects and lights
def your_own_scene():

    sphere_a = Sphere([-0.2, 0, -1], 0.8)
    sphere_a.set_material([1, 1, 0], [0, 1, 0], [0.7, 1, 0.6], 50, 0.8)
    triangle = Rectangle([1, -1, -2], [0, 1, -1.5], [0, -1, -1], [-2, -1.5, 1])
    triangle.set_material([1, 0, 0], [1, 1, 0], [0, 0, 0], 100, 1)
    plane = Plane([0, 0, 1], [0, 0, -3])
    plane.set_material([0, 0, 1], [0, 1, 0], [1, 1, 1], 70, 0.2)

    light_a = SpotLight(intensity=np.array([0.5, 0.5, 0.9]), position=np.array([0.8, 0.7, 0]), direction=([0, 0, 1]),
                        kc=0.3, kl=0.7, kq=0.1)
    light_b = PointLight(intensity=np.array([0.8, 1, 0.5]), position=np.array([1, 2, 0]), kc=0.1, kl=0.5, kq=0.7)

    camera = np.array([0, 0, 1])
    lights = [light_a, light_b]

    objects = [sphere_a, triangle, plane]
    # please use this ambient
    # ambient = np.array([0.1, 0.2, 0.3])
    return camera,lights, objects
