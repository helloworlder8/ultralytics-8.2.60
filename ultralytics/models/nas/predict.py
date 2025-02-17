# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class NASPredictor(BasePredictor):
    """
    Ultralytics YOLO NAS Predictor for object detection.

    This class extends the `BasePredictor` from Ultralytics engine and is responsible for post-processing the
    raw predictions generated by the YOLO NAS models. It applies operations like non-maximum suppression and
    scaling the bounding boxes to fit the original image dimensions.

    Attributes:
        args (Namespace): Namespace containing various configurations for post-processing.

    Example:
        ```python
        from ultralytics import NAS

        model = NAS('yolo_nas_s')
        predictor = model.predictor
        # Assumes that raw_preds, img, orig_imgs are available
        results = predictor.postprocess(raw_preds, img, orig_imgs)
        ```

    Note:
        Typically, this class is not instantiated directly. It is used internally within the `NAS` class.
    """

    def postprocess(self, preds_in, img, orig_imgs):
        """Postprocess predictions and returns a list of Results objects."""

        # Cat boxes and class scores
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)

        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.NMS_Threshold,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
