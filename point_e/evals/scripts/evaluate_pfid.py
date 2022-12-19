"""
Evaluate P-FID between two batches of point clouds.

The point cloud batches should be saved to two npz files, where there
is an arr_0 key of shape [N x K x 3], where K is the dimensionality of
each point cloud and N is the number of clouds.
"""

import argparse

from point_e.evals.feature_extractor import PointNetClassifier, get_torch_devices
from point_e.evals.fid_is import compute_statistics
from point_e.evals.npz_stream import NpzStreamer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("batch_1", type=str)
    parser.add_argument("batch_2", type=str)
    args = parser.parse_args()

    print("creating classifier...")
    clf = PointNetClassifier(devices=get_torch_devices(), cache_dir=args.cache_dir)

    print("computing first batch activations")

    features_1, _ = clf.features_and_preds(NpzStreamer(args.batch_1))
    stats_1 = compute_statistics(features_1)
    del features_1

    features_2, _ = clf.features_and_preds(NpzStreamer(args.batch_2))
    stats_2 = compute_statistics(features_2)
    del features_2

    print(f"P-FID: {stats_1.frechet_distance(stats_2)}")


if __name__ == "__main__":
    main()
