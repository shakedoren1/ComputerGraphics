from typing import Optional
import numpy as np
from abc import abstractmethod

INF = np.inf
SHIFT_CONST = 1e-5


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal


## Lights
class LightSource:

    def __init__(self, intensity: np.array):
        self.intensity = intensity

    @abstractmethod
    def get_light_ray(self, intersection):
        pass


class DirectionalLight(LightSource):

    def __init__(self, intensity: np.array, direction: np.array):
        super().__init__(intensity)
        self.direction = np.array(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection_point: np.array):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return INF

    # This function returns the light intensity at a point
    # the intensity is constant in all points
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):

    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / self.get_attenuation_factor(d)

    def get_attenuation_factor(self, d):
        return self.kc + self.kl * d + self.kq * (d ** 2)


class SpotLight(LightSource):

    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq
        self.direction = np.array(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        v = normalize(self.position - intersection)
        return (self.intensity * (np.dot(v, self.direction))) / self.get_attenuation_factor(d)

    def get_attenuation_factor(self, d):
        return self.kc + self.kl * d + self.kq * (d ** 2)


class Ray:
    def __init__(self, origin: np.array, direction: np.array):
        self.origin = origin
        self.direction = direction

    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        """
        This function iterates over all objects in the scene and looks for the one with minimum distance.
        because the ray is represented as ray = origin + t * direction, we want to find the object that its t is minimal,
        because t in parametric representation represent the distance between point to origin
        :param objects : List of Object3D of all the objects in the scene
        """
        min_t = np.inf
        min_distance_object = None
        for obj in objects:
            t, scene_object = obj.intersect(self)
            if not t or (t < 0):
                continue
            elif 0 < t < min_t:
                min_t = t
                min_distance_object = scene_object
        if not min_distance_object:
            return None, None
        return min_t, min_distance_object

    def get_intersection_point(self, t):
        return self.origin + t * self.direction


class Object3D:

    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = np.array(ambient)  # ka
        self.diffuse = np.array(diffuse)  # kd
        self.specular = np.array(specular)  # ks
        self.shininess = shininess  # n
        self.reflection = reflection  # kr

    @abstractmethod
    def intersect(self, ray: Ray):
        """
        This function checks if the 3d object intersects with a Ray = origin + direction * t
        if its intersect its returns the scalar t that satisfies this equation
        """
        pass


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None, None

    def get_normal_to_surface(self, intersection_point):
        return normalize(self.normal)

class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |  
         |               |  
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.d = np.array(d)
        self.ab = self.b - self.a
        self.ad = self.d - self.a 
        self.bc = self.c - self.b
        self.normal = self.compute_normal()

    def compute_normal(self):
        n = np.cross(self.ab, self.ad)
        return normalize(n)

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        rectangle_plane = Plane(self.normal, self.a)
        r, plane = rectangle_plane.intersect(ray)
        if not r:
            return None, None
        # else, check if the intersection point inside the rectangle
        point_of_intersection = ray.get_intersection_point(r)
        is_point_in_rectangle = self.point_in_rectangle(point_of_intersection)
        if is_point_in_rectangle:
            return r, self
        return None, None
    
    def point_in_rectangle(self, p):
        pa = self.a - p
        pb = self.b - p
        pc = self.c - p
        pd = self.d - p
        pa_pb = np.dot(np.cross(pa, pb), self.normal)
        pb_pc = np.dot(np.cross(pb, pc), self.normal)
        pc_pd = np.dot(np.cross(pc, pd),self.normal)
        pd_pa = np.dot(np.cross(pd, pa),self.normal)
        return pa_pb>0 and pb_pc>0 and pc_pd>0 and pd_pa>0


    def get_normal_to_surface(self, intersection_point):
        return self.normal



class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """ 
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """
        A = B = C = D = E = F = None
        g = np.array(f) + (np.array(a)-np.array(d))
        h = np.array(e) + (np.array(b)-np.array(c))
        A = Rectangle (a,b,c,d)
        B = Rectangle (d,c,e,f)
        C = Rectangle (f,e,h,g)
        D = Rectangle (g,h,b,a)
        E = Rectangle (a,d,f,g)
        F = Rectangle (b,c,e,h)
        self.face_list = [A,B,C,D,E,F]


    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        n_dist = np.inf
        n_obj = self.face_list[0]
        for rec in self.face_list:
            inter, p = rec.intersect(ray)
            if inter is not None:
                if inter < n_dist:
                    n_dist = inter
                    n_obj = rec
        if n_dist != np.inf:
            return n_dist, n_obj
        return None, None
        

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        # a = 1 cause -> a = ||ray_direction||^2 but -> ||ray_direction|| = 1 (we used normalized vector)
        a = 1
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            # if one of t is negative it means that the ray that intersects the sphere have negative direction vector which means for example that the sphere is behind the camera and the screen
            if t1 > 0 and t2 > 0:
                t = min(t1, t2)
                return t, self
        return None, None

    def get_normal_to_surface(self, intersection_point):
        return normalize(intersection_point - self.center)
