import math
import cv2
import numpy as np
import random
# Original synthetic dataset module

class DrawShape:
    def __init__(self, width, height, shapes = ['line', 'ellipse', 'polygon', 'star', 'cube']):
        self.width = width
        self.height = height
        self.shapes = shapes

    def __call__(self, num_objects):
        edge_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        fill_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        all_points = []

        for _ in range(num_objects):
            shape = random.choice(self.shapes)
            if shape == 'line':
                points = self.draw_line(edge_mask, fill_mask)
            elif shape == 'ellipse':
                points = self.draw_ellipse(edge_mask, fill_mask)
            elif shape == 'polygon':
                points = self.draw_polygon(edge_mask, fill_mask)
            elif shape == 'star':
                points = self.draw_star(edge_mask, fill_mask)
            else:  # cube
                points = self.draw_cube(edge_mask, fill_mask)
            all_points.extend(points)
            # points.append(shape(edge_mask, fill_mask))

        return edge_mask, fill_mask, all_points
    
    def draw_line(self, edge_mask, fill_mask, thickness=2):
        """Draw a random line"""
        x1 = random.randint(0, self.width)
        y1 = random.randint(0, self.height)
        x2 = random.randint(0, self.width)
        y2 = random.randint(0, self.height)

        # Draw edge
        cv2.line(edge_mask, (x1, y1), (x2, y2), 255, thickness)

        # Draw thicker line for fill
        cv2.line(fill_mask, (x1, y1), (x2, y2), 255, thickness * 2)

        return [(x1, y1), (x2, y2)]


    def draw_ellipse(self, edge_mask, fill_mask):
        """Draw a random ellipse"""
        center_x = random.randint(self.width // 4, 3 * self.width // 4)
        center_y = random.randint(self.height // 4, 3 * self.height // 4)

        # Random axes lengths
        max_axis = min(self.width, self.height) // 4
        axes = (random.randint(max_axis // 2, max_axis),
                random.randint(max_axis // 2, max_axis))

        angle = random.uniform(0, 360)

        # Draw edge
        cv2.ellipse(edge_mask, (center_x, center_y), axes, angle, 0, 360, 255, 2)

        # Draw filled ellipse
        cv2.ellipse(fill_mask, (center_x, center_y), axes, angle, 0, 360, 255, -1)

        # Generate points around ellipse for keypoints
        points = []
        for t in np.linspace(0, 2 * np.pi, 32):
            x = int(center_x + axes[0] * np.cos(t) * np.cos(np.deg2rad(angle)) -
                    axes[1] * np.sin(t) * np.sin(np.deg2rad(angle)))
            y = int(center_y + axes[0] * np.cos(t) * np.sin(np.deg2rad(angle)) +
                    axes[1] * np.sin(t) * np.cos(np.deg2rad(angle)))
            points.append((x, y))
        return points


    def draw_polygon(self, edge_mask, fill_mask, max_sides=8):
        """Draw a random polygon"""
        num_corners = random.randint(3, max_sides)
        min_dim = min(self.height, self.width)
        rad = int(max(random.random() * min_dim / 2, min_dim / 10))

        # Center point
        x = random.randint(int(rad), int(self.width - rad))
        y = random.randint(int(rad), int(self.height - rad))

        # Generate points
        slices = np.linspace(0, 2 * math.pi, num_corners + 1)
        points = []
        for i in range(num_corners):
            angle = slices[i] + random.random() * (slices[i + 1] - slices[i])
            px = int(x + max(random.random(), 0.4) * rad * math.cos(angle))
            py = int(y + max(random.random(), 0.4) * rad * math.sin(angle))
            points.append((px, py))

        points = np.array(points)

        # Draw edges
        for i in range(len(points)):
            pt1 = points[i]
            pt2 = points[(i + 1) % len(points)]
            cv2.line(edge_mask, pt1, pt2, 255, 2)

        # Draw fill
        cv2.fillPoly(fill_mask, [points], 255)

        return points.tolist()


    def draw_star(self, edge_mask, fill_mask, nb_branches=6):
        """Draw a random star"""
        num_branches = random.randint(3, nb_branches)
        min_dim = min(self.height, self.width)
        rad = int(max(random.random() * min_dim / 2, min_dim / 5))

        # Center point
        x = random.randint(rad, self.width - rad)
        y = random.randint(rad, self.height - rad)

        # Generate points
        points = []
        center_point = (x, y)

        slices = np.linspace(0, 2 * math.pi, num_branches + 1)
        for i in range(num_branches):
            angle = slices[i] + random.random() * (slices[i + 1] - slices[i])
            px = int(x + max(random.random(), 0.3) * rad * math.cos(angle))
            py = int(y + max(random.random(), 0.3) * rad * math.sin(angle))
            points.append((px, py))

            # Draw edges
            cv2.line(edge_mask, center_point, (px, py), 255, 2)
            cv2.line(fill_mask, center_point, (px, py), 255, 4)

        points.insert(0, center_point)
        return points


    def draw_cube(self, edge_mask, fill_mask):
        """Draw a random cube"""
        min_dim = min(self.height, self.width)
        min_size = min_dim * 0.2

        # Generate cube dimensions
        lx = min_size + random.random() * min_dim / 3
        ly = min_size + random.random() * min_dim / 3
        lz = min_size + random.random() * min_dim / 3

        # Create cube points
        cube = np.array([
            [0, 0, 0], [lx, 0, 0], [0, ly, 0], [lx, ly, 0],
            [0, 0, lz], [lx, 0, lz], [0, ly, lz], [lx, ly, lz]
        ])

        # Apply random rotation
        angles = [random.uniform(math.pi / 10, 4 * math.pi / 10) for _ in range(3)]
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(angles[0]), -math.sin(angles[0])],
            [0, math.sin(angles[0]), math.cos(angles[0])]
        ])
        rot_y = np.array([
            [math.cos(angles[1]), 0, math.sin(angles[1])],
            [0, 1, 0],
            [-math.sin(angles[1]), 0, math.cos(angles[1])]
        ])
        rot_z = np.array([
            [math.cos(angles[2]), -math.sin(angles[2]), 0],
            [math.sin(angles[2]), math.cos(angles[2]), 0],
            [0, 0, 1]
        ])

        # Apply transformations
        cube = cube @ rot_x @ rot_y @ rot_z

        # Move to center of image
        cube = cube + np.array([self.width / 2, self.height / 2, 0])
        points_2d = cube[:, :2].astype(int)

        # Define visible faces
        faces = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])

        # Draw edges
        for face in faces:
            for i in range(len(face)):
                pt1 = tuple(points_2d[face[i]])
                pt2 = tuple(points_2d[face[(i + 1) % len(face)]])
                cv2.line(edge_mask, pt1, pt2, 255, 2)

        # Draw fills
        for face in faces:
            face_points = points_2d[face]
            cv2.fillPoly(fill_mask, [face_points], 255)

        return points_2d[1:].tolist()  # Exclude hidden corner