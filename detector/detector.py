import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from torchvision import transforms
from shapely import Polygon, MultiPolygon


class BoundingBox:
    """class representing a 2D bounding box
    """

    def __init__(self, xyxy: list[float]):
        """
        Initialization

        Args:
            xyxy (list[float]): float values representing the bounding box coordinates 
                in the format [x_min, y_min, x_max, y_max], which will be rounded to integers
        """
        self.top = round(xyxy[1])
        self.left = round(xyxy[0])
        self.bottom = round(xyxy[3])
        self.right = round(xyxy[2])

    def tl(self):
        """
        Returns the top-left coordinates of the bounding box

        Returns:
            tuple: left and top coordinates
        """
        return (self.left, self.top)

    def br(self):
        """Returns the bottom-right coordinates of the bounding box

        Returns:
            tuple: right and bottom coordinates
        """
        return (self.right, self.bottom)


class FineSegmentation:
    """class for fine-grained semantic segmentation of image patches using a pre-trained model
    """
    NORM_FACTOR = 32  # normalization factor for input image size

    def __init__(self, model_path: str, mean: tuple[float, float, float], std_dev: tuple[float, float, float]):
        """Loads the pre-trained model and initializes the image transformation

        Args:
            model_path (str): path to the pre-trained model
            mean (tuple[float, float, float]): mean values for color normalization
            std_dev (tuple[float, float, float]): standard-deviation values for color normalization
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std_dev)])

    def apply(self, image: np.array) -> np.array:
        """Applies the segmentation model to the input image

        Args:
            image (np.array): input image in BGR format

        Returns:
            np.array: single-channel segmentation mask with the same size as the input image
        """
        # resize image to the nearest multiple of NORM_FACTOR and apply transformations
        h, w = image.shape[:2]
        input_image = cv2.resize(image, [self.NORM_FACTOR * round(d / self.NORM_FACTOR) for d in [w, h]])
        input_image = self.transform(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)).to(self.device)
        # apply the model and return the segmentation mask
        with torch.no_grad():
            input_image = input_image.unsqueeze(0)
            output = self.model(input_image)
            assert len(output.shape) == 4 and output.shape[1] > 1, \
                f'Unexpected output shape: {output.shape}. Expected shape (N, C, H, W) with C > 1.'
            pred_mask = torch.argmax(output, dim=1).cpu().squeeze(0)
        return cv2.resize(np.array(pred_mask), (w, h), interpolation=cv2.INTER_NEAREST)


class ObjectInstance:
    """class representing an object instance detected in the image
    """

    def __init__(self, box: BoundingBox, mask: np.array, contour: Polygon, confidence: float, area: float = None):
        """
        Initialization

        Args:
            box (BoundingBox): bounding box of the object
            mask (np.array): single-channel semantic-segmentation mask of the object with same size as the bounding box
            contour (Polygon): contour of the object in original image coordinates
            confidence (float): confidence score of the original detection
            area (float, optional): absolute area of the object
        """
        self.box = box
        self.mask = mask
        self.contour = contour
        self.confidence = confidence
        self.area = area


class Detector:
    """class for detecting and segmenting objects in images by first delineating their overall shapes using 
        instance segmentation and then applying fine-grained semantic segmentation within each contour. 
        Additionally, object masses can be estimated based on the detection of reference markers.
    """

    def __init__(self, coarse_model_path: str, fine_model_path: str, marker_model_path: str,
                 mean: tuple[float, float, float], std_dev: tuple[float, float, float], marker_areas: dict[int, float]):
        """Loads the pre-trained models

        Args:
            coarse_model_path (str): path to coarse-grained instance-segmentation model
            fine_model_path (str): path to fine-grained semantic-segmentation model
            marker_model_path (str): path to marker-detection model for reference objects
            mean (tuple[float, float, float]): mean values for color normalization
            std_dev (tuple[float, float, float]): standard-deviation values for color normalization
            marker_areas (dict[int, float]): dictionary mapping marker labels to their absolute areas in mm^2
        """
        self.coarse_model = YOLO(coarse_model_path)
        assert self.coarse_model.model.model[-1].nc == 1, \
            f'Expected 1 class in coarse model, but found {self.coarse_model.model.model[-1].nc} classes.'
        self.fine_model = FineSegmentation(fine_model_path, mean, std_dev)
        self.marker_model = None if marker_model_path is None else YOLO(marker_model_path)
        self.marker_areas = marker_areas

    def apply(self, image: np.array,
              min_confidence: float = 0.4, marker_min_confidence=0.4) -> tuple[list[ObjectInstance], list[Polygon]]:
        """Applies the detection and segmentation models to the input image

        Args:
            image (np.array): input image in BGR format
            min_confidence (float, optional): minimum confidence for coarse-grained instance segmentation
            marker_min_confidence (float, optional): minimum confidence for marker detection

        Returns:
            list[ObjectInstance]: list of detected object instances
            list[Polygon]: list of detected markers
        """
        scale, markers = self.infer_scale(image, marker_min_confidence)
        instances = []
        for res in self.coarse_model.predict(image, verbose=False, retina_masks=True, conf=min_confidence)[0]:
            box, contour = self.extract_contour(res)
            if box is not None and contour is not None:
                seg = self.fine_model.apply(image[box.top:box.bottom, box.left:box.right])
                instances.append(ObjectInstance(box, seg, contour, res.boxes.conf[0],
                                                None if scale is None else scale * contour.area))
        return instances, markers

    @staticmethod
    def extract_contour(instance: Results) -> tuple[BoundingBox, Polygon]:
        """Extracts the bounding box and contour from a detected instance

        Args:
            instance (Results): detected instance

        Returns:
            BoundingBox: bounding box of the object
            Polygon: contour of the object in original image coordinates
        """
        box = BoundingBox(instance.boxes.xyxy[0].tolist())
        contour_pts = instance.masks.xy[0]
        if len(contour_pts) > 0:
            contour = Polygon(contour_pts)
            if not contour.is_valid:
                contour = contour.buffer(0)
                if isinstance(contour, MultiPolygon):
                    contour = max(list(contour.geoms), key=lambda c: c.length)
            if isinstance(contour, Polygon) and contour.is_valid:
                return box, contour
        print('\tWarning: invalid object contour!')
        return None, None

    def infer_scale(self, image: np.array, min_confidence) -> tuple[float, list[Polygon]]:
        """Infers the absolute scale of objects using detections of markers in the image

        Args:
            image (np.array): input image in BGR format
            min_confidence (float): minimum confidence for marker detection

        Returns:
            float: scale factor for the detected objects
            list[Polygon]: list of detected markers
        """
        if self.marker_model is not None:
            results = self.marker_model.predict(image, verbose=False, conf=min_confidence)[0].obb
            if len(results) > 0:
                marker_obbs = [Polygon(b) for b in results.xyxyxyxy.tolist()]
                return np.average([self.marker_areas[label] / marker_obb.area
                                  for label, marker_obb in zip(results.cls.tolist(), marker_obbs)
                                  if label in self.marker_areas]), marker_obbs
        return None, []
