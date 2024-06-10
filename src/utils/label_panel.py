import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DraggableRectangle:
    def __init__(self, rect, text, label, ax):
        self.rect = rect
        self.text = text
        self.label = label
        self.ax = ax
        self.press = None
        self.background = None
        self.corner_selected = None
        self.corners = self.create_corners()

    def create_corners(self):
        # Create corners for resizing
        x, y = self.rect.xy
        w, h = self.rect.get_width(), self.rect.get_height()
        corners = {
            "bottom_left": patches.Circle((x, y), 5, color='blue', picker=True),
            "bottom_right": patches.Circle((x + w, y), 5, color='blue', picker=True),
            "top_left": patches.Circle((x, y + h), 5, color='blue', picker=True),
            "top_right": patches.Circle((x + w, y + h), 5, color='blue', picker=True)
        }
        for corner in corners.values():
            self.ax.add_patch(corner)
        return corners

    def connect(self):
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.rect.axes:
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            # Check if a corner is selected
            for key, corner in self.corners.items():
                contains, attrd = corner.contains(event)
                if contains:
                    self.corner_selected = key
                    break
            if not self.corner_selected:
                return
        self.press = (self.rect.xy, (event.xdata, event.ydata))
        self.rect.set_animated(True)
        self.text.set_animated(True)
        self.rect.figure.canvas.draw()
        self.background = self.rect.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.rect)
        self.ax.draw_artist(self.text)
        for corner in self.corners.values():
            corner.set_animated(True)
            self.ax.draw_artist(corner)
        self.rect.figure.canvas.blit(self.ax.bbox)

    def on_motion(self, event):
        if self.press is None:
            return
        if event.inaxes != self.rect.axes:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        if self.corner_selected:
            # Resize the rectangle
            if self.corner_selected == "bottom_left":
                self.rect.set_x(x0 + dx)
                self.rect.set_y(y0 + dy)
                self.rect.set_width(self.rect.get_width() - dx)
                self.rect.set_height(self.rect.get_height() - dy)
            elif self.corner_selected == "bottom_right":
                self.rect.set_y(y0 + dy)
                self.rect.set_width(self.rect.get_width() + dx)
                self.rect.set_height(self.rect.get_height() - dy)
            elif self.corner_selected == "top_left":
                self.rect.set_x(x0 + dx)
                self.rect.set_width(self.rect.get_width() - dx)
                self.rect.set_height(self.rect.get_height() + dy)
            elif self.corner_selected == "top_right":
                self.rect.set_width(self.rect.get_width() + dx)
                self.rect.set_height(self.rect.get_height() + dy)
        else:
            # Move the rectangle
            self.rect.set_x(x0 + dx)
            self.rect.set_y(y0 + dy)

        self.update_corners()
        self.update_text()

        self.rect.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.rect)
        self.ax.draw_artist(self.text)
        for corner in self.corners.values():
            self.ax.draw_artist(corner)
        self.rect.figure.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        self.press = None
        self.corner_selected = None
        self.rect.set_animated(False)
        self.text.set_animated(False)
        for corner in self.corners.values():
            corner.set_animated(False)
        self.background = None
        self.rect.figure.canvas.draw()

    def update_corners(self):
        x, y = self.rect.xy
        w, h = self.rect.get_width(), self.rect.get_height()
        self.corners["bottom_left"].center = (x, y)
        self.corners["bottom_right"].center = (x + w, y)
        self.corners["top_left"].center = (x, y + h)
        self.corners["top_right"].center = (x + w, y + h)

    def update_text(self):
        x, y = self.rect.xy
        self.text.set_position((x, y))

    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

    def get_coordinates(self):
        x, y = self.rect.xy
        w, h = self.rect.get_width(), self.rect.get_height()
        return self.label, x, y, x + w, y + h


def visualize_predictions(image_path, predictions, output_file="bounding_boxes.txt"):
    """Plots bounding boxes and scores on an image with draggable rectangles and saves to file.

    Args:
        image_path: Path to the image file.
        predictions: Dictionary containing "boxes" and "scores". If "labels"
                     is present, it should have the same length as "boxes".
        output_file: File to save the bounding box coordinates.
    """

    # Load the image
    image = cv2.imread(image_path)

    # Convert predictions to numpy arrays
    boxes = predictions["boxes"].detach().numpy()
    scores = predictions["scores"].detach().numpy()
    labels = (
        predictions["labels"].detach().numpy()
        if "labels" in predictions
        else [None] * len(boxes)
    )  # Handle optional labels

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    draggables = []

    # Iterate over detected objects
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box

        # Draw bounding box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="red", linewidth=2
        )
        ax.add_patch(rect)

        # Display score and label (optional)
        text_items = []
        if score is not None:
            text_items.append(f"{score:.2f}")
        if label is not None:
            text_items.append(str(label))
        text = ", ".join(text_items) if text_items else ""
        text = ax.text(
            xmin,
            ymin,
            text,
            fontsize=10,
            bbox=dict(facecolor="yellow", alpha=0.5),
        )

        # Make the rectangle draggable
        draggable = DraggableRectangle(rect, text, label, ax)
        draggable.connect()
        draggables.append(draggable)

    plt.title("Object Detection Results")
    plt.axis("off")
    plt.show()

    # Save the coordinates to a file
    with open(output_file, 'w') as f:
        for draggable in draggables:
            label, xmin, ymin, xmax, ymax = draggable.get_coordinates()
            f.write(f"{label} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n")


if __name__ == "__main__":
    # Example Usage (replace with your actual paths and model)
    image_path = "data/valid/images/000002_jpg.rf.XyqagWMEa65XqGa5uLPK.jpg"

    # Replace this with your model's actual prediction output
    predictions = {
        "boxes": torch.tensor([[100, 150, 250, 300], [50, 80, 180, 220]]),
        "labels": torch.tensor([1, 3]),  # Corrected labels
        "scores": torch.tensor([0.85, 0.92]),
    }

    visualize_predictions(image_path, predictions, "bounding_boxes.txt")
