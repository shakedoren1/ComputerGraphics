import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
import functools

    
def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        # Calculate grayscale image by performing matrix multiplication
        return np.dot(np_img, self.gs_weights)
        
        
    # @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        def calc_pixel_energy(padded_gs, x, y):
            """ Calculate energy of a pixel

            Parameters:
                x (int): horizontal index of the pixel
                y (int): vertical index of the pixel

            Returns:
                energy of the pixel (float32)

            Guidelines & hints:
                - The energy of a pixel is the sum of the absolute value of the gradient of the pixel
            """
            horizontal_gradient = padded_gs[x, y+1] - padded_gs[x, y]
            vertical_gradient = padded_gs[x+1, y] - padded_gs[x, y]
            return np.sqrt(np.square(horizontal_gradient) + np.square(vertical_gradient))
    
        # Pad boundaries with 0.5 to prevent outlier values
        padded_gs = np.pad(self.resized_gs, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0.5)
        
        E = np.zeros_like(self.resized_gs)[:,:,0].astype(float)
        
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                E[i, j] = calc_pixel_energy(padded_gs,i, j)
        
        return E
    
        # Calculate horizontal gradient
        horizontal_gradient = np.roll(self.gs, -1, axis=1) - self.gs

        # Calculate vertical gradient
        vertical_gradient = np.roll(self.gs, -1, axis=0) - self.gs

        # Compute energy matrix
        energy_matrix = np.sqrt(horizontal_gradient ** 2 + vertical_gradient ** 2)
        
        # Map [-1,1] gradient values to grayscale values [0,1]
        # energy_matrix = (energy_matrix + 1) / 2
        
        print(np.array_equal(energy_matrix,energy_matrix_1))
        return energy_matrix

        
    def calc_M_bt(self):
        pass
             
    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise):
        pass

    def init_mats(self):
        pass

    def update_ref_mat(self):
        for i, s in enumerate(self.seam_history[-1]):
            self.idx_map[i, s:] += 1

    def backtrack_seam(self):
        pass

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0

        
        
class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.M, self.backtrack_mat = VerticalSeamImage.calc_M_bt(self.E)
        except NotImplementedError as e:
            print(e)
        
        
    @staticmethod
    @jit(nopython=True)
    def calc_M_bt(E):
        """ Calculates the matrix M discussed in lecture (with forward-looking cost)

        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """            
        def calc_M_ij(M, E, i,j):
                has_left_neighbor = j > 0
                has_right_neighbor =  j < E.shape[1] - 1
                
                if has_left_neighbor and has_right_neighbor:
                    vert_cost = M[i-1,j] + np.abs(E[i,j-1] - E[i,j+1])
                    left_cost = M[i-1,j-1] + np.abs(E[i,j-1] - E[i-1,j])
                    right_cost = M[i-1,j+1] + np.abs(E[i,j+1] - E[i-1,j])
                
                elif has_right_neighbor:
                    vert_cost = M[i-1,j]
                    left_cost = np.inf
                    right_cost = M[i-1,j+1] + np.abs(E[i,j+1] - E[i-1,j])
                    
                else: # has_left_neighbor
                    vert_cost = M[i-1,j]
                    left_cost = M[i-1,j-1] + np.abs(E[i,j-1] - E[i-1,j])
                    right_cost = np.inf
                
                if vert_cost <= left_cost and vert_cost <= right_cost:
                    idx_to_bt = [i-1, j]
                elif left_cost <= vert_cost and left_cost <= right_cost:
                    idx_to_bt = [i-1, j-1]
                else:
                    idx_to_bt = [i-1, j+1]

                return E[i,j] + np.minimum(vert_cost, np.minimum(left_cost,right_cost)), idx_to_bt

        M = np.zeros_like(E, dtype=np.float32)
        backtrack_mat = np.zeros((E.shape[0], E.shape[1], 2),dtype=np.int64)
        
        M[0, :] = E[0, :]
        
        for i in range(1, M.shape[0]):
            for j in range(M.shape[1]):
                M[i, j], backtrack_mat[i,j] = calc_M_ij(M, E, i, j)
                
        return M, backtrack_mat
        
        padded_gs = np.pad(self.gs, pad_width=1, mode='constant', constant_values=np.inf)

        # Calculate the gradients for C_V, C_L, and C_R
        gradient_V = np.abs(np.roll(padded_gs, -1, axis=0) - np.roll(padded_gs, 1, axis=0))
        gradient_L = np.abs(np.roll(padded_gs, -1, axis=1) - np.roll(padded_gs, 1, axis=1)) + \
                    np.abs(np.roll(padded_gs, -1, axis=0) - np.roll(padded_gs, -1, axis=1))
        gradient_R = np.abs(np.roll(padded_gs, -1, axis=1) - np.roll(padded_gs, 1, axis=1)) + \
                    np.abs(np.roll(padded_gs, 1, axis=1) - np.roll(padded_gs, -1, axis=0))

        # Initialize the matrix M with zeros
        M = np.zeros_like(self.E, dtype=np.float32)
        padded_M = np.copy(padded_gs)

        # Calculate the energy for the first row
        M[0, :] = self.E[0, :]
        padded_M[1, 1:self.w] = self.E[0, :]

        # Iterate over each row in the image
        for i in range(1, self.h):
            # Calculate the value of c_V, c_L, and c_R for the current row
            c_V = gradient_V[i, 1:self.w]
            c_L = gradient_L[i, 1:self.w]
            c_R = gradient_R[i, 1:self.w]
            print(c_V.shape)

        
            # Calculate M(i, j) based on the given formula
            M[i, :] = self.E[i, :] + np.minimum(padded_M[i - 1, 1:self.w] + c_V,
                                                np.minimum(np.roll(padded_M[i - 1, 1:self.w], -1, axis=1) + c_R,
                                                        np.roll(padded_M[i - 1, 1:self.w], 1, axis=1) + c_L))
            
            padded_M[i, 1:self.w] = M[i, :]

        return M


    # @NI_decor
    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """
        self.init_mats()
        for n in range(num_remove):
            # Find minimum cell in bottom row in M
            bottom_idx_seam = np.argmin(self.M[-1])
            
            # backtrack this cell up to top row - get a list of indices seam_idx
            seam_idx = self.backtrack_seam([self.M.shape[0]-1, bottom_idx_seam])
            
            # update self.idx_map_h, self.idx_map_v
            self.update_idx_maps(seam_idx)
                        
            # delete these seam_idx pixels from self.resized_rgb
            self.remove_seam()
            
            # update self.E and self.M efficiently
            self.init_mats()
            
            print(f"removed seam {n+1}, shape is now: ", self.resized_gs.shape)
            # add seam_idx to seam_history
            self.seam_history.append((n, seam_idx))
                                
################ DEBUG ##################################### DEBUG ###########################
    def test_update_idx_maps(self):
        size = 5

        # Initialize idx_map_h
        idx_map_h = np.tile(np.arange(size), (size, 1))

        # Initialize idx_map_v
        idx_map_v = np.tile(np.arange(size).reshape(-1, 1), (1, size))

        print(idx_map_h)
        print(idx_map_v)
        
        seam_idx = [(4, 3), (3, 3), (2, 3), (1, 2), (0, 1)]
        self.update_idx_maps(idx_map_h, idx_map_v, seam_idx)
        
        print(idx_map_h)
        print(idx_map_v)
################ DEBUG ##################################### DEBUG ###########################

    def update_mats(self):
        pass
        
    def update_idx_maps(self,seam_idx):
        # Iterate through each index (i0, j0) in seam_idx
        for i0, j0 in seam_idx:
            # Update horizontal idx map
            self.idx_map_h[i0, j0:] += 1
            
            # Update vertical idx map
            #  if hor - rotate v clockwise and for each row increment until j
            # self.idx_map_v[i0:, j0] += 1

    def paint_seams(self):
        for s in self.seam_history:
            for i, s_i in enumerate(s):
                self.cumm_mask[self.idx_map_v[i,s_i], self.idx_map_h[i,s_i]] = False
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])

    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.M, self.backtrack_mat = VerticalSeamImage.calc_M_bt(self.E)
        self.mask = np.ones_like(self.M, dtype=bool)

    # @NI_decor
    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.rotate_mats(clockwise=True)
        self.seams_removal(num_remove)
        self.rotate_mats(clockwise=False)

    # @NI_decor
    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.seams_removal(num_remove)

    # @NI_decor
    def backtrack_seam(self, idx):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        i, j = idx
        seam_idx = [idx]
        for r in range(self.M.shape[0] - 1, 1, -1):
            next_step = self.backtrack_mat[i,j].tolist()
            seam_idx.insert(0, next_step)
            i, j = next_step
        
        return seam_idx
            

    # @NI_decor
    def remove_seam(self):
        """ Removes a seam from self.rgb (you may create a resized version, like self.resized_rgb)

        Guidelines & hints:
        In order to apply the removal, you might want to extend the seam mask to support 3 channels (rgb) using: 3d_mak = np.stack([1d_mask] * 3, axis=2), and then use it to create a resized version.
        """
        # create mask array
        h, w = self.M.shape
        mask = np.ones((h, w), dtype=bool)
        
        # find minimum value in M's last row
        j = np.argmin(self.M[-1])
        
        # Remove pixels specified by seam_idx from resized_gs
        for i in reversed(range(h)):
            mask[i, j] = False
            j = self.backtrack_mat[i, j].tolist()[1]

        self.resized_gs = self.resized_gs[mask].reshape(h, w - 1, 1)
        
    # @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")
    
    # @NI_decor
    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    # @NI_decor
    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    # @NI_decor
    @staticmethod
    # @jit(nopython=True)
    def calc_bt(M, E, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.
        
        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        h, w = M.shape
    
        
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")
    
class SCWithObjRemoval(VerticalSeamImage):
    def __init__(self, active_masks=['Gemma'], *args, **kwargs):
        import glob
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        self.active_masks = active_masks
        self.obj_masks = {basename(img_path)[:-4]: self.load_image(img_path, format='L') for img_path in glob.glob('images/obj_masks/*')}

        try:
            self.preprocess_masks()
        except KeyError:
            print("TODO (Bonus): Create and add Jurassic's mask")
        
        try:
            self.M = self.calc_M_bt()
        except NotImplementedError as e:
            print(e)

    def preprocess_masks(self):
        """ Mask preprocessing.
            different from images, binary masks are not continous. We have to make sure that every pixel is either 0 or 1.

            Guidelines & hints:
                - for every active mask we need make it binary: {0,1}
        """
        raise NotImplementedError("TODO: Implement SeamImage.preprocess_masks")
        print('Active masks:', self.active_masks)

    # @NI_decor
    def apply_mask(self):
        """ Applies all active masks on the image
            
            Guidelines & hints:
                - you need to apply the masks on other matrices!
                - think how to force seams to pass through a mask's object..
        """
        raise NotImplementedError("TODO: Implement SeamImage.apply_mask")


    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M_bt()
        self.apply_mask() # -> added
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)
        self.mask = np.ones_like(self.M, dtype=bool)

    def reinit(self, active_masks):
        """ re-initiates instance
        """
        self.__init__(active_masks=active_masks, img_path=self.path)

    def remove_seam(self):
        """ A wrapper for super.remove_seam method. takes care of the masks.
        """
        super().remove_seam()
        for k in self.active_masks:
            self.obj_masks[k] = self.obj_masks[k][self.mask].reshape(self.h, self.w)


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


