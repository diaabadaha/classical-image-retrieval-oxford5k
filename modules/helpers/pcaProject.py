from pathlib import Path
import numpy as np
import helpers.ioStore as ioStore
from sklearn.decomposition import PCA  # type: ignore


def fitAndSavePca(databaseVectors, pcaDim):
    outputDir = Path("dataset/pca")
    ioStore.ensureDir(outputDir)

    pca = PCA(n_components=pcaDim, random_state=42)
    reduced = pca.fit_transform(databaseVectors).astype(np.float32)

    # Normalize after PCA (important for cosine)
    reduced = np.vstack([ioStore.l2Normalize(v) for v in reduced]).astype(np.float32)

    np.savez_compressed(
        str(outputDir / f"pca_{pcaDim}.npz"),
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
    )
    ioStore.saveNpy(outputDir / f"vbowPca_{pcaDim}.npy", reduced)

    print(f"Saved PCA model: {outputDir / f'pca_{pcaDim}.npz'}")
    print(
        f"Saved PCA vectors: {outputDir / f'vbowPca_{pcaDim}.npy'}  shape={reduced.shape}"
    )


def loadPcaModel(pcaDim):
    p = Path("dataset/pca") / f"pca_{pcaDim}.npz"
    data = np.load(str(p))
    return data["mean"].astype(np.float32), data["components"].astype(np.float32)


def applyPcaToVector(vec, mean, components):
    # vec: (K,)
    centered = (vec - mean).astype(np.float32)
    reduced = (components @ centered).astype(np.float32)  # (pcaDim,)
    return ioStore.l2Normalize(reduced)


def loadPcaDatabaseVectors(pcaDim):
    p = Path("dataset/pca") / f"vbowPca_{pcaDim}.npy"
    return ioStore.loadNpy(p).astype(np.float32)
