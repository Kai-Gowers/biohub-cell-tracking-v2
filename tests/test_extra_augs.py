"""Coordinate/target consistency of the opt-in extra augmentations in pack_train."""
import numpy as np
import torch

from cell_tracking.pack_train import (
    flip_augment, intensity_augment, noise_augment, rot90_xy_augment, time_reverse_window,
)


def _spike_window(points, shape=(2, 8, 16, 16)):
    """One bright voxel per (frame, z, y, x) point; coords padded to M=4."""
    imgs = torch.zeros(shape)
    coords = torch.zeros(shape[0], 4, 3)
    masks = torch.zeros(shape[0], 4, dtype=torch.bool)
    fill = [0] * shape[0]
    for (t, z, y, x) in points:
        imgs[t, z, y, x] = 1.0
        coords[t, fill[t]] = torch.tensor([z, y, x], dtype=torch.float32)
        masks[t, fill[t]] = True
        fill[t] += 1
    return imgs, coords, masks


def _spikes_match(imgs, coords, masks):
    for t in range(imgs.shape[0]):
        for i in torch.nonzero(masks[t]).flatten():
            z, y, x = coords[t, i].long().tolist()
            assert imgs[t, z, y, x] == 1.0, (t, z, y, x)


def test_rot90_moves_coords_with_image():
    imgs, coords, masks = _spike_window([(0, 3, 2, 11), (1, 5, 9, 4), (1, 0, 0, 0)])
    for seed in range(20):
        rng = np.random.default_rng(seed)
        out, c, m = rot90_xy_augment(imgs.clone(), coords, masks, rng=rng)
        assert out.shape == imgs.shape
        _spikes_match(out, c, m)
        assert (c[~m] == 0).all()           # padding untouched
        assert torch.equal(coords, _spike_window([(0, 3, 2, 11), (1, 5, 9, 4), (1, 0, 0, 0)])[1])  # input not mutated


def test_rot90_composes_with_flip():
    imgs, coords, masks = _spike_window([(0, 3, 2, 11), (1, 5, 9, 4)])
    for seed in range(20):
        rng = np.random.default_rng(seed)
        out, c, m = flip_augment(imgs, coords, masks, rng=rng)
        out, c, m = rot90_xy_augment(out, c, m, rng=rng)
        _spikes_match(out, c, m)


def test_rot90_skips_non_square():
    imgs, coords, masks = _spike_window([(0, 3, 2, 11)], shape=(2, 8, 16, 12))
    rng = np.random.default_rng(1)
    out, c, m = rot90_xy_augment(imgs, coords, masks, rng=rng)
    assert out.shape == imgs.shape and torch.equal(c, coords)


def test_intensity_and_noise_leave_coords_alone():
    imgs, coords, masks = _spike_window([(0, 3, 2, 11)])
    imgs = imgs + 0.2
    rng = np.random.default_rng(0)
    out, c, m = intensity_augment(imgs, coords, masks, rng=rng, gain_range=(0.5, 2.0), gamma_range=(0.5, 2.0))
    assert torch.isfinite(out).all() and out.min() >= 0 and torch.equal(c, coords) and torch.equal(m, masks)
    out2, c, m = noise_augment(out, coords, masks, rng=rng, std_max=0.05)
    assert out2.shape == imgs.shape and torch.equal(c, coords)
    assert abs(float((out2 - out).std()) - 0.0) < 0.06


def test_time_reverse_transposes_targets_and_skips_divisions():
    imgs, coords, masks = _spike_window([(0, 3, 2, 11), (0, 1, 1, 1), (1, 5, 9, 4)])
    targets = torch.zeros(1, 4, 4)
    targets[0, 0, 0] = 1.0  # node 0 at t -> node 0 at t+1
    rng = np.random.default_rng(3)  # first draw > 0.5?  force both branches by looping seeds
    seen_rev = seen_id = False
    for seed in range(20):
        rng = np.random.default_rng(seed)
        i2, c2, m2, t2 = time_reverse_window(imgs, coords, masks, targets, rng=rng)
        if torch.equal(i2, imgs):
            seen_id = True
            continue
        seen_rev = True
        assert torch.equal(i2, imgs.flip(0)) and torch.equal(c2, coords.flip(0)) and torch.equal(m2, masks.flip(0))
        assert torch.equal(t2[0], targets[0].T)
        _spikes_match(i2, c2, m2)
    assert seen_rev and seen_id
    # a division (one source, two targets) must block the reversal
    div = torch.zeros(1, 4, 4); div[0, 0, 0] = 1.0; div[0, 0, 1] = 1.0
    for seed in range(20):
        i2, c2, m2, t2 = time_reverse_window(imgs, coords, masks, div, rng=np.random.default_rng(seed))
        assert torch.equal(i2, imgs) and torch.equal(t2, div)
