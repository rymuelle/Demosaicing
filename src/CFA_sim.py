import numpy as np
from PIL import Image

def simulate_sparse_PIL(image, **kwargs):
    arr = np.array(image).transpose(2, 0, 1)
    return simulate_sparse(arr, **kwargs)[0].transpose(1, 2, 0)
    return Image.fromarray(simulate_sparse(arr, **kwargs)[0].transpose(1, 2, 0))

def simulate_sparse_wrapper(image, **kwargs):
    arr = image.transpose(2, 0, 1)
    return simulate_sparse(arr, **kwargs)[0].transpose(1, 2, 0)



def simulate_sparse(image, pattern="RGGB", cfa_type="bayer", bias = 255):
    """
    Simulate a sparse CFA (Color Filter Array) from an RGB image.

    Args:
        image: numpy array (3, H, W), RGB image in [0, 1] or [0, 255].
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer,
                 or ignored if cfa_type="xtrans".
        cfa_type: "bayer" or "xtrans".

    Returns:
        cfa: numpy array (r, H, W), sparse CFA image.
        sparse_mask:  numpy array (r, H, W), mask of pixels.
    """
    _, H, W= image.shape
    cfa = np.zeros((3, H, W), dtype=image.dtype) - bias
    sparse_mask = np.zeros((3, H, W), dtype=image.dtype)
    if cfa_type == "bayer":
        # 2×2 Bayer masks
        masks = {
            "RGGB": np.array([["R", "G"], ["G", "B"]]),
            "BGGR": np.array([["B", "G"], ["G", "R"]]),
            "GRBG": np.array([["G", "R"], ["B", "G"]]),
            "GBRG": np.array([["G", "B"], ["R", "G"]]),
        }
        if pattern not in masks:
            raise ValueError(f"Unknown Bayer pattern: {pattern}")

        mask = masks[pattern]
        cmap = {"R": 0, "G": 1, "B": 2}
         
        for i in range(2):
            for j in range(2):
                ch = cmap[mask[i, j]]
                cfa[ch, i::2, j::2] = image[ch, i::2, j::2]
                sparse_mask[ch, i::2, j::2] = 1
    elif cfa_type == "xtrans":
        # Fuji X-Trans 6×6 repeating pattern
        xtrans_pattern = np.array([
            ["G","B","R","G","R","B"],
            ["R","G","G","B","G","G"],
            ["B","G","G","R","G","G"],
            ["G","R","B","G","B","R"],
            ["B","G","G","R","G","G"],
            ["R","G","G","B","G","G"],
        ])
        cmap = {"R":0, "G":1, "B":2}

        for i in range(6):
            for j in range(6):
                ch = cmap[xtrans_pattern[i, j]]
                cfa[ch, i::6, j::6] = image[ch, i::6, j::6]
                sparse_mask[ch, i::2, j::2] = 1
    else:
        raise ValueError(f"Unknown CFA type: {cfa_type}")

    return cfa, sparse_mask



def cfa_to_sparse(image, pattern="RGGB", cfa_type="bayer"):
    """
    Make a sparse representation from a CFA

    Args:
        image: numpy array (H, W), RGB image in [0, 1] or [0, 255].
        pattern: CFA pattern string, one of {"RGGB","BGGR","GRBG","GBRG"} for Bayer,
                 or ignored if cfa_type="xtrans".
        cfa_type: "bayer" or "xtrans".

    Returns:
        cfa: numpy array (r, H, W), sparse CFA image.
        sparse_mask:  numpy array (r, H, W), mask of pixels.
    """
    H, W= image.shape
    cfa = np.zeros((3, H, W), dtype=image.dtype)
    sparse_mask = np.zeros((3, H, W), dtype=image.dtype)
    if cfa_type == "bayer":
        # 2×2 Bayer masks
        masks = {
            "RGGB": np.array([["R", "G"], ["G", "B"]]),
            "BGGR": np.array([["B", "G"], ["G", "R"]]),
            "GRBG": np.array([["G", "R"], ["B", "G"]]),
            "GBRG": np.array([["G", "B"], ["R", "G"]]),
        }
        if pattern not in masks:
            raise ValueError(f"Unknown Bayer pattern: {pattern}")

        mask = masks[pattern]
        cmap = {"R": 0, "G": 1, "B": 2}
         
        for i in range(2):
            for j in range(2):
                ch = cmap[mask[i, j]]
                cfa[ch, i::2, j::2] = image[i::2, j::2]
                sparse_mask[ch, i::2, j::2] = 1
    elif cfa_type == "xtrans":
        # Fuji X-Trans 6×6 repeating pattern
        xtrans_pattern = np.array([
            ["G","B","R","G","R","B"],
            ["R","G","G","B","G","G"],
            ["B","G","G","R","G","G"],
            ["G","R","B","G","B","R"],
            ["B","G","G","R","G","G"],
            ["R","G","G","B","G","G"],
        ])
        cmap = {"R":0, "G":1, "B":2}

        for i in range(6):
            for j in range(6):
                ch = cmap[xtrans_pattern[i, j]]
                cfa[ch, i::6, j::6] = image[i::6, j::6]
                sparse_mask[ch, i::2, j::2] = 1
    else:
        raise ValueError(f"Unknown CFA type: {cfa_type}")

    return cfa, sparse_mask

