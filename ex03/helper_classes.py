import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hits
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    v = np.array([0,0,0])
    v = vector - 2 * np.dot(vector, axis) * axis
    return v

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = - normalize(np.array(direction))

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, self.direction)
        
    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, _):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, _):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = - normalize(np.array(direction))
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        v = normalize(self.position - intersection)
        return (self.intensity * (np.dot(v, self.direction))) / (self.kc + self.kl * d + self.kq * (d ** 2))


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf
        for obj in objects:
            result = obj.intersect(self)
            if result is not None:
                t, obj = result
                if t is not None and t < min_distance:
                    nearest_object = obj
                    min_distance = t
                    intersections = self.origin + t * self.direction
        return nearest_object, min_distance, intersections


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def get_normal(self, _):
        return normalize(self.normal)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        return normalize(np.cross(self.b - self.a, self.c - self.b))

    def get_normal(self, _):
        return normalize(self.normal)

    # This function checks if a ray intersects the triangle
    def intersect(self, ray: Ray):
        p = Plane(self.normal, self.a)
        result = p.intersect(ray)
        if result is None:
            return None
        t, _ = result
        intersection = ray.origin + t * ray.direction
        if self.is_inside(intersection):
            return t, self
        else:
            return None
        
    # This function checks if a point is inside the triangle
    def is_inside(self, point):
        # Convert point to a numpy array if it isn't one
        P = np.array(point)
        
        # Vectors from triangle points to the intersection point
        AP = P - self.a
        BP = P - self.b
        CP = P - self.c
        
        # Calculate the degree of the angle between the vectors
        angel_AP_PB = np.pi - np.arccos(np.dot(AP, -BP) / (np.linalg.norm(AP) * np.linalg.norm(-BP)))
        angel_BP_PC = np.pi - np.arccos(np.dot(BP, -CP) / (np.linalg.norm(BP) * np.linalg.norm(-CP)))
        angel_CP_PA = np.pi - np.arccos(np.dot(CP, -AP) / (np.linalg.norm(CP) * np.linalg.norm(-AP)))

        # Check if the point is inside the triangle
        if np.isclose(angel_AP_PB + angel_BP_PC + angel_CP_PA, 2*np.pi):
            return True
        return False


class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        for idx in t_idx:
            l.append(Triangle(self.v_list[idx[0]], self.v_list[idx[1]], self.v_list[idx[2]]))
        return l

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    # This function checks if a ray intersects the pyramid
    def intersect(self, ray: Ray):
        t = np.inf
        nearest_obj = None
        for triangle in self.triangle_list:
            result = triangle.intersect(ray)
            if result is not None:
                t_obj, obj = result
                if t_obj < t:
                    t = t_obj
                    nearest_obj = obj
        if nearest_obj is None:
            return None
        return t, nearest_obj

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def get_normal(self, point):
        return normalize(point - self.center)

    # This function checks if a ray intersects the sphere ####### Needs to go over!!!!!!
    def intersect(self, ray: Ray):
        # Let R(t) be the point in the ray for t, and S be the sphere equation
        # by comparing R(t) = S we isolate t and get a quadratic equation
        # let L = ray origin - sphere center, D is ray direction
        # t^2 (D D) + 2t (L D) + (L L - r^2) = 0
        L = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = np.dot(L, ray.direction) * 2.0
        c = np.dot(L, L) - (self.radius ** 2)
        
        # compute the discriminant
        disc = b * b - 4 * a * c
        
        if disc < 0:
            return None
        
        sqrtDisc = np.sqrt(disc)
        t1 = (-b - sqrtDisc) / (2.0 * a)
        t2 = (-b + sqrtDisc) / (2.0 * a)
        
        # return the closer intersection, or None if t1 and t2 are negative
        if t1 < 0:
            t1 = t2
        if t1 < 0:
            return None
        return t1, self
