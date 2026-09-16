import json
from pathlib import Path
import numpy as np


def ensureDir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def listAllImages(imagesDir=Path("dataset/images")):
    paths = []
    for p in Path(imagesDir).rglob("*"):
        if p.is_file():
            paths.append(p)
    paths.sort()
    return paths


def listAllPreprocessedImages(preprocessedDir=Path("dataset/preprocessed")):
    paths = []
    for p in Path(preprocessedDir).rglob("*"):
        if p.is_file():
            paths.append(p)
    paths.sort()
    return paths


def listAllDescriptorFiles(descriptorsDir=Path("dataset/descriptors")):
    paths = []
    for p in Path(descriptorsDir).rglob("*"):
        if p.is_file() and p.suffix.lower() == ".npy":
            paths.append(p)
    paths.sort()
    return paths


def loadJson(jsonPath):
    jsonPath = Path(jsonPath)
    if not jsonPath.exists():
        return {}
    with jsonPath.open("r", encoding="utf-8") as f:
        return json.load(f)


def saveJson(jsonPath, obj):
    jsonPath = Path(jsonPath)
    ensureDir(jsonPath.parent)
    with jsonPath.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def loadNpy(npyPath):
    npyPath = Path(npyPath)
    if not npyPath.exists():
        raise FileNotFoundError(f"Missing file: {npyPath}")
    return np.load(str(npyPath))


def saveNpy(npyPath, arr):
    npyPath = Path(npyPath)
    ensureDir(npyPath.parent)
    np.save(str(npyPath), arr)


def loadGroundTruth(groundTruthPath=Path("dataset/groundtruth.json")):
    return loadJson(groundTruthPath)


def loadCodebook(codebookPath=Path("dataset/codeBook/codeBook.npy")):
    centers = loadNpy(codebookPath).astype(np.float32)
    if centers.ndim != 2 or centers.shape[1] != 128:
        raise RuntimeError(f"Invalid codeBook shape: {centers.shape}")
    return centers


def loadIdf(idfPath=Path("dataset/vbow/idf.npy")):
    idf = loadNpy(idfPath).astype(np.float32)
    if idf.ndim != 1:
        raise RuntimeError(f"Invalid idf shape: {idf.shape}")
    return idf


def loadDatabaseVectors(vbowPath=Path("dataset/vbow/vbowTfidf.npy")):
    mat = loadNpy(vbowPath).astype(np.float32)
    if mat.ndim != 2:
        raise RuntimeError(f"Invalid vbowTfidf shape: {mat.shape}")
    return mat


def loadImageNames(namesPath=Path("dataset/vbow/imageNames.json")):
    names = loadJson(namesPath)
    if not isinstance(names, list):
        raise RuntimeError("imageNames.json must be a list of filenames")
    return names


def loadDescriptorsForImage(imageName, descriptorsDir=Path("dataset/descriptors")):
    descPath = Path(descriptorsDir) / (Path(imageName).stem + ".npy")
    if not descPath.exists():
        return np.zeros((0, 128), dtype=np.float32)

    desc = np.load(str(descPath)).astype(np.float32)
    if desc.ndim != 2 or desc.shape[1] != 128:
        return np.zeros((0, 128), dtype=np.float32)
    return desc


def loadKeypointsForImage(imageName, keypointsDir=Path("dataset/keypoints")):
    kpPath = Path(keypointsDir) / (Path(imageName).stem + ".npy")
    if not kpPath.exists():
        return np.zeros((0, 2), dtype=np.float32)

    kp = np.load(str(kpPath)).astype(np.float32)
    if kp.ndim != 2 or kp.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return kp


def l2Normalize(vec):
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)
