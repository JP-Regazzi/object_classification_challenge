import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance

class ColorAnalysis:
    def extract_object_colors(self, image_path, num_colors=3):
        """
        Extract the most dominant colors from an image, ignoring the white background.
        Returns dominant colors in Lab space.
        """
        # Load image and convert to HSV
        image = cv2.imread(image_path)
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask to filter out white background (Saturation low & Brightness high)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 30, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Invert mask to get object only
        object_mask = cv2.bitwise_not(mask)

        # Extract object pixels
        object_pixels = cv2.bitwise_and(image, image, mask=object_mask)

        # Convert to RGB
        object_pixels = cv2.cvtColor(object_pixels, cv2.COLOR_BGR2RGB)

        # Reshape and remove black pixels (caused by masking)
        pixels = object_pixels.reshape(-1, 3)
        pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]  # Remove black pixels

        if len(pixels) == 0:
            return None  # No valid pixels found

        # Find most frequent colors
        pixels_tuple = [tuple(p) for p in pixels]
        most_common_colors = [color for color, _ in Counter(pixels_tuple).most_common(num_colors)]

        # Convert to Lab space for better color comparison
        most_common_colors_lab = cv2.cvtColor(
            np.array([most_common_colors], dtype=np.uint8), cv2.COLOR_RGB2Lab
        )[0]

        return most_common_colors_lab

    def color_distance(self, color_set_1, color_set_2):
        """
        Compute the minimum Euclidean distance between two sets of Lab colors.
        """
        return min(distance.euclidean(c1, c2) for c1 in color_set_1 for c2 in color_set_2)

    def filter_images_by_color(self, query_image_path, database_folder, threshold=30, num_colors=3):
        """
        Filters images in a database based on color similarity.
        Returns a list of image paths that are within the color similarity threshold.
        """
        query_colors = self.extract_object_colors(query_image_path, num_colors=num_colors)
        if query_colors is None:
            return []

        matching_images = []
        count = 0

        for img_file in os.listdir(database_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
                img_path = os.path.join(database_folder, img_file)

                db_colors = self.extract_object_colors(img_path, num_colors=num_colors)
                if db_colors is None:
                    continue

                # Compute color similarity
                if self.color_distance(query_colors, db_colors) < threshold:
                    matching_images.append(img_path)

                # Simple progress log
                if count % 100 == 0:
                    print(f"Processed {count} images...")

        return matching_images

    def load_cv2_image(self, path):
        """
        Load an image using OpenCV and convert BGR to RGB for matplotlib.
        """
        img_cv2 = cv2.imread(path)
        if img_cv2 is None:
            return None
        return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    def save_color_info_to_csv(self, database_folder, csv_path, num_colors=3):
        """
        For each image in `database_folder`, extract the dominant colors (in Lab),
        and save them to a CSV at `csv_path`.

        The CSV will have columns:
           [filename, L1, a1, b1, L2, a2, b2, ..., L<num_colors>, a<num_colors>, b<num_colors>]
        """
        all_rows = []

        for img_file in os.listdir(database_folder):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(database_folder, img_file)
                colors_lab = self.extract_object_colors(img_path, num_colors=num_colors)
                if colors_lab is None:
                    continue

                # Flatten Lab values (shape: (num_colors, 3) -> list of length 3*num_colors)
                flattened_colors = colors_lab.flatten().tolist()

                # Prepare row: [filename, L1, a1, b1, L2, a2, b2, ...]
                row = [img_file] + flattened_colors
                all_rows.append(row)

        # Create column names
        columns = ['filename']
        for i in range(1, num_colors+1):
            columns += [f'L{i}', f'a{i}', f'b{i}']

        df = pd.DataFrame(all_rows, columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"Color info CSV saved to: {csv_path}")

    def filter_images_by_color_from_csv(self, query_image_path, csv_path,
                                        database_folder, threshold=30, num_colors=3):
        """
        Loads color info from CSV and filters images by color similarity with the query image.
        Returns a list of matching image paths.
        """
        # Extract colors from the query image
        query_colors = self.extract_object_colors(query_image_path, num_colors=num_colors)
        if query_colors is None:
            return []

        # Load CSV into a DataFrame
        df = pd.read_csv(csv_path)

        matching_images = []
        for _, row in df.iterrows():
            # Reconstruct Lab color set from CSV
            colors_flat = []
            for i in range(1, num_colors+1):
                l = row[f'L{i}']
                a = row[f'a{i}']
                b = row[f'b{i}']
                colors_flat.append([l, a, b])

            db_colors = np.array(colors_flat, dtype=np.float32)

            # Compute color similarity
            dist = self.color_distance(query_colors, db_colors)
            if dist < threshold:
                # Reconstruct full path from folder + filename
                img_path = os.path.join(database_folder, row['filename'])
                matching_images.append(img_path)

        return matching_images

    def plot_query_and_matches(self, query_image, filtered_images):
        """
        Plots the query image and matching images in a grid.
        Parameters:
            query_image (str): Path to the query image file.
            filtered_images (list of str): List of paths to matched images.
        """
        num_images = len(filtered_images)

        if num_images > 0:
            # Max 5 images per row (query + 4 matches)
            cols = min(num_images + 1, 5)
            rows = math.ceil((num_images + 1) / cols)

            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axs = np.array(axs).reshape(-1)  # Flatten in case of multiple rows

            # Show the query image in the first subplot
            query_img_rgb = self.load_cv2_image(query_image)
            if query_img_rgb is not None:
                axs[0].imshow(query_img_rgb)
                axs[0].set_title("Query Image", fontsize=12, fontweight="bold")
            axs[0].axis("off")

            # Show filtered images in the subsequent subplots
            for i, img_file in enumerate(filtered_images, start=1):
                match_img_rgb = self.load_cv2_image(img_file)
                if match_img_rgb is not None:
                    axs[i].imshow(match_img_rgb)
                    axs[i].set_title(f"Match {i}", fontsize=10)
                axs[i].axis("off")

            # Hide any unused subplots
            for j in range(i + 1, len(axs)):
                axs[j].axis("off")

            plt.tight_layout()
            plt.show()
        else:
            print("No filtered images to display.")


if __name__ == "__main__":

    database_folder = "./data/DAM"
    csv_path = "./data/color_database.csv"
    query_image_path = "./data/preprocessed_test/image-20210928-102713-12d2869d.jpg"
    threshold = 30
    num_colors = 3

    # Create an instance of the class
    color_analyzer = ColorAnalysis()

    # 1) Save color info into a CSV (you can skip this if already done)
    color_analyzer.save_color_info_to_csv(database_folder, csv_path, num_colors=num_colors)

    # 2) Filter images using the newly created CSV
    matching_images = color_analyzer.filter_images_by_color_from_csv(
        query_image_path=query_image_path,
        csv_path=csv_path,
        database_folder=database_folder,
        threshold=threshold,
        num_colors=num_colors
    )

    # 3) Plot the query image and any matching images
    color_analyzer.plot_query_and_matches(query_image_path, matching_images)
