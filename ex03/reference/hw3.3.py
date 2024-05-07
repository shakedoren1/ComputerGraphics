from helper_classes import *
import matplotlib.pyplot as plt

# epsilon = 10 ** -6

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
#             color = np.zeros(3, dtype='float64')

            # This is the main loop where each pixel color is computed.
            # TODO
            # Ray: p(t) = p0 + tv
            
            nearest_object, min_t = ray.nearest_intersected_object(objects)

            hit_point = (ray.origin + min_t * ray.direction)
            if isinstance(nearest_object, Sphere):
                normal = nearest_object.get_normal(hit_point)
            else:
                normal = nearest_object.normal
            
            
            hit_point += 1e-4 * normalize(normal)
            color = get_color(ambient, objects, lights, hit_point, nearest_object, camera, 0, max_depth)
            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


def get_color(ambient, objects, lights, hit_point, hit_object, origin, level, max_depth):
    color = np.zeros(3)
#     color = np.zeros(3, dtype='float64')
    
    color = calc_ambient_color(hit_object, ambient)
        
    for light_source in lights:
        
        # Shadow ray
        ray_intersection_to_light = light_source.get_light_ray(hit_point)
        nearest_object, t_value_to_light = ray_intersection_to_light.nearest_intersected_object(objects)
        if isinstance(light_source, DirectionalLight):
            blocked = nearest_object is not None and t_value_to_light > 0
        else:
#             blocked = t_value_to_light < light_source.get_distance_from_light(hit_point)
            blocked = t_value_to_light > 0 and t_value_to_light < light_source.get_distance_from_light(hit_point)
        
        if not blocked:
#             color += calc_diffuse_color(hit_object, hit_point, light_source, ray_intersection_to_light) + calc_specular_color(hit_object, hit_point, light_source, ray_intersection_to_light, origin)
#             print("color type =" + str(type(color)))
#             print("color val =" + str(color))
            diffuse_val = calc_diffuse_color(hit_object, hit_point, light_source, ray_intersection_to_light)
#             print("diffuse val type= " + str(type(diffuse_val)))
#             print("diffuse_val val =" + str(diffuse_val))
            specular_val = calc_specular_color(hit_object, hit_point, light_source, ray_intersection_to_light, origin)
#             print("specular_val type= " + str(type(specular_val)))
#             print("specular_val val =" + str(specular_val))
            
            color = color.astype('float64') + diffuse_val + specular_val
            
            
    level = level + 1

    # Changed > to >=
    if level >= max_depth:
        return color

    reflective_ray = constructReflectiveRay(ray_intersection_to_light, hit_point, hit_object)
    nearest_object, min_t = reflective_ray.nearest_intersected_object(objects)
    if nearest_object != None and nearest_object.reflection > 0:
        reflective_hit_point = reflective_ray.origin + min_t * reflective_ray.direction
        if isinstance(nearest_object, Sphere):
            normal = nearest_object.get_normal(reflective_hit_point)
        else:
            normal = nearest_object.normal
            
        reflective_hit_point += 1e-4 * normal
        
        color += nearest_object.reflection * get_color(ambient, objects, lights, reflective_hit_point, nearest_object, origin, level, max_depth)

    return color
    
# Reflective calculations
        
def calc_diffuse_color(hit_object, hit_point, light_source, light_ray):
    # Added normalization to light_ray.direction
#     return hit_object.diffuse * light_source.intensity * (hit_object.normal.dot(normalize(light_ray.direction)))
    if isinstance(hit_object, Sphere):
        normal = hit_object.get_normal(hit_point)
    else:
        normal = hit_object.normal
        
    normal = normalize(normal)
    
    return hit_object.diffuse * light_source.get_intensity(hit_point) * (normal.dot(normalize(light_ray.direction)))

    
    
def constructReflectiveRay(ray_intersection_to_light, hit_point, hit_object):
    ray_dir = normalize(ray_intersection_to_light.direction)
    if isinstance(hit_object, Sphere):
        normal = hit_object.get_normal(hit_point)
    else:
        normal = hit_object.normal
        
#     normal = normalize(normal)

    reflective_ray_dir = ray_dir - 2 * (ray_dir.dot(normal)) * normal

    return Ray(hit_point, -reflective_ray_dir)

def calc_specular_color(hit_object, hit_point, light_source, light_ray, origin):
    
    if isinstance(hit_object, Sphere):
        normal = hit_object.get_normal(hit_point)
    else:
        normal = hit_object.normal
    
    normal = normalize(normal)
    
    reflection_direction = normalize(light_ray.direction - 2 * (light_ray.direction.dot(normal) * normal))
    viewer_direction = normalize(origin - hit_point)
    return hit_object.specular * light_source.get_intensity(hit_point) * (reflection_direction.dot(viewer_direction) ** hit_object.shininess)

    
    
def calc_ambient_color(hit_object, ambient):
    return hit_object.ambient * ambient

# Write your own objects and lights
# TODO
def your_own_scene():
#     camera = np.array([0,0,1])
#     lights = []
#     objects = []
#     return camera, lights, objects

    kernel = Sphere([0, -0.2, -0.2], 0.4)
    kernel.set_material([0.8, 0, 1], [0, 1, 0], [0.4, 0.4, 0.4], 200, 2)
    plane = Plane([0, 1, 0], [0, -0.3, 0])
    plane.set_material([0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [1, 1, 1], 1000, 0.5)
    background = Plane([0, 0, 1], [0, 0, -3])
    background.set_material([0.4, 0.2, 0.8], [0.4, 0.2, 0.8], [0.4, 0.2, 0.8], 1000, 0.5)
    objects = [kernel, plane, background]

    pointlight = PointLight(intensity=np.array([0.7, 0.7, 0.7]), position=np.array([1, 1.5, 1]), kc=0.1, kl=0.1, kq=0.1)
    spotlight = SpotLight(intensity=np.array([1, 1, 1]), position=np.array([1, 1, 0]), direction=([0, -0.5, 0.3]),
                          kc=0.1, kl=0.1, kq=0.1)
    lights = [pointlight, spotlight]
    ambient = np.array([0.1, 0.2, 0.3])
    camera = np.array([0, 0, 1])

    return camera, lights, objects, ambient
