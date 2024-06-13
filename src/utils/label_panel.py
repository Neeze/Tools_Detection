import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import TextBox

class DraggableRectangle:
    def __init__(self, rect, text, label, ax, color, update_label_callback):
        self.rect = rect
        self.text = text
        self.label = label
        self.ax = ax
        self.press = None
        self.background = None
        self.corner_selected = None
        self.corners = self.create_corners()
        self.color = color
        self.update_label_callback = update_label_callback

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
        # Call the update label callback
        self.update_label_callback(self)

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

    def update_label(self, new_label):
        self.label = new_label
        self.text.set_text(new_label)
        self.text.figure.canvas.draw()

def visualize_predictions(image:torch.Tensor, predictions, output_file="bounding_boxes.txt"):
    """Plots bounding boxes and scores on an image with draggable rectangles and saves to file.

    Args:
        image_path: Path to the image file.
        predictions: Dictionary containing "boxes" and "scores". If "labels"
                     is present, it should have the same length as "boxes".
        output_file: File to save the bounding box coordinates.
    """

    # # Load the image
    # image = cv2.imread(image_path)

    # Convert predictions to numpy arrays
    boxes = predictions["boxes"].detach().cpu().numpy()
    scores = predictions["scores"].detach().cpu().numpy()
    labels = (
        predictions["labels"].detach().cpu().numpy()
        if "labels" in predictions
        else [None] * len(boxes)
    )  # Handle optional labels

    colors = ['red', 'green', 'blue', 'purple', 'orange']  # Define colors for different labels

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    draggables = []
    selected_rectangle = [None]  # Store the currently selected rectangle

    # Update label callback function
    def update_label(draggable):
        selected_rectangle[0] = draggable

    # Iterate over detected objects
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box

        # Assign a color based on the label
        color = colors[label % len(colors)] if label is not None else 'red'

        # Draw bounding box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2
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
        draggable = DraggableRectangle(rect, text, label, ax, color, update_label)
        draggable.connect()
        draggables.append(draggable)

    # TextBox for changing labels
    axbox = plt.axes([0.15, 0.01, 0.2, 0.05])  # Adjust position and size as needed
    text_box = TextBox(axbox, 'New Label: ')

    def submit_label(text):
        if selected_rectangle[0]:
            selected_rectangle[0].update_label(text)
            selected_rectangle[0].rect.set_edgecolor(colors[int(text) % len(colors)])

    text_box.on_submit(submit_label)

    # Function to add a new rectangle
    def add_rectangle(event):
        if event.key == 'a' and event.inaxes == ax:
            x, y = event.xdata, event.ydata
            rect = patches.Rectangle((x, y), 0, 0, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            text = ax.text(x, y, "", fontsize=10, bbox=dict(facecolor="yellow", alpha=0.5))
            draggable = DraggableRectangle(rect, text, 0, ax, 'red', update_label) # Default label is 0
            draggable.connect()
            draggables.append(draggable)
            selected_rectangle[0] = draggable
            fig.canvas.draw()
    
    # Function to confirm/update a rectangle
    def confirm_rectangle(event):
        if event.key == 'enter' and selected_rectangle[0] is not None:
            selected_rectangle[0] = None  
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', add_rectangle)
    fig.canvas.mpl_connect('key_press_event', confirm_rectangle)

    plt.title("Object Detection Results")
    plt.axis("off")
    plt.show()

    # Save the coordinates to a file
    with open(output_file, 'w') as f:
        for draggable in draggables:
            label, xmin, ymin, xmax, ymax = draggable.get_coordinates()
            f.write(f"{label} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n")

if __name__ == "__main__":
    image_path = "data/valid/images/000002_jpg.rf.XyqagWMEa65XqGa5uLPK.jpg"

    predictions = {
        "boxes": torch.tensor([[100, 150, 250, 300], [50, 80, 180, 220], [225.16, 133.23, 375.16, 283.23]]),
        "labels": torch.tensor([1, 3, 9]), 
        "scores": torch.tensor([0.85, 0.92, 0.2]),
    }

    visualize_predictions(image_path, predictions, "labels.txt")
