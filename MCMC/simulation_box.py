import numpy as np

class SimulationBox:
    def __init__(self, box_size_x, box_size_y=None):
        """
        Create a 2D simulation box with given x and y box lengths.

        If box_size_y is not provided, we assume a square box of size_x = size_y.
        """
        if box_size_y is None:
            box_size_y = box_size_x  # square by default

        self.box_size_x = box_size_x
        self.box_size_y = box_size_y

        # In 2D, 'volume' is actually the area:
        self.volume = self.box_size_x * self.box_size_y

    def apply_pbc(self, position, checking=False):
        """
        Apply 2D periodic boundary conditions to the given position.
        """
        wrapped_position = np.array([
            position[0] % self.box_size_x,
            position[1] % self.box_size_y
        ])
        if checking:
            print("position % (box_size_x, box_size_y) =", wrapped_position)
        return wrapped_position

    def minimum_image(self, position_1, position_2, checking=False):
        """
        Compute the minimum-image displacement vector for 2D rectangular PBC.
        """
        delta = position_1 - position_2

        # Wrap each dimension independently
        delta[0] -= self.box_size_x * np.round(delta[0] / self.box_size_x)
        delta[1] -= self.box_size_y * np.round(delta[1] / self.box_size_y)

        if checking:
            print("position_1 =", position_1)
            print("position_2 =", position_2)
            print("position_1 - position_2 =", position_1 - position_2)
            print("wrapped_delta =", delta)
        return delta

    def compute_distance(self, position_1, position_2, checking=False):
        """
        Return the minimum-image Euclidean distance between two 2D positions.
        """
        min_image = self.minimum_image(position_1, position_2, checking=checking)
        distance = np.linalg.norm(min_image)
        if checking:
            print("np.linalg.norm(min_image) =", distance)
        return distance
    
    def compute_distances(self, position_1, positions_2, checking=False):
        """
        Compute the minimum-image distances from position_1 to an array of positions.
        """
        distances = np.zeros(len(positions_2))
        for i, position_2 in enumerate(positions_2):
            distances[i] = self.compute_distance(position_1, position_2, checking=checking)
        return distances
