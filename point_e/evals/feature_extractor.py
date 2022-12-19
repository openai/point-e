from abc import ABC, abstractmethod
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from point_e.models.download import load_checkpoint

from .npz_stream import NpzStreamer
from .pointnet2_cls_ssg import get_model


def get_torch_devices() -> List[Union[str, torch.device]]:
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        return ["cpu"]


class FeatureExtractor(ABC):
    @property
    @abstractmethod
    def supports_predictions(self) -> bool:
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @abstractmethod
    def features_and_preds(self, streamer: NpzStreamer) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a stream of point cloud batches, compute feature vectors and class
        predictions.

        :param point_clouds: a streamer for a sample batch. Typically, arr_0
                             will contain the XYZ coordinates.
        :return: a tuple (features, predictions)
                 - features: a [B x feature_dim] array of feature vectors.
                 - predictions: a [B x num_classes] array of probabilities.
        """


class PointNetClassifier(FeatureExtractor):
    def __init__(
        self,
        devices: List[Union[str, torch.device]],
        device_batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ):
        state_dict = load_checkpoint("pointnet", device=torch.device("cpu"), cache_dir=cache_dir)[
            "model_state_dict"
        ]

        self.device_batch_size = device_batch_size
        self.devices = devices
        self.models = []
        for device in devices:
            model = get_model(num_class=40, normal_channel=False, width_mult=2)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            self.models.append(model)

    @property
    def supports_predictions(self) -> bool:
        return True

    @property
    def feature_dim(self) -> int:
        return 256

    @property
    def num_classes(self) -> int:
        return 40

    def features_and_preds(self, streamer: NpzStreamer) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = self.device_batch_size * len(self.devices)
        point_clouds = (x["arr_0"] for x in streamer.stream(batch_size, ["arr_0"]))

        output_features = []
        output_predictions = []

        with ThreadPool(len(self.devices)) as pool:
            for batch in point_clouds:
                batch = normalize_point_clouds(batch)
                batches = []
                for i, device in zip(range(0, len(batch), self.device_batch_size), self.devices):
                    batches.append(
                        torch.from_numpy(batch[i : i + self.device_batch_size])
                        .permute(0, 2, 1)
                        .to(dtype=torch.float32, device=device)
                    )

                def compute_features(i_batch):
                    i, batch = i_batch
                    with torch.no_grad():
                        return self.models[i](batch, features=True)

                for logits, _, features in pool.imap(compute_features, enumerate(batches)):
                    output_features.append(features.cpu().numpy())
                    output_predictions.append(logits.exp().cpu().numpy())

        return np.concatenate(output_features, axis=0), np.concatenate(output_predictions, axis=0)


def normalize_point_clouds(pc: np.ndarray) -> np.ndarray:
    centroids = np.mean(pc, axis=1, keepdims=True)
    pc = pc - centroids
    m = np.max(np.sqrt(np.sum(pc**2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / m
    return pc
