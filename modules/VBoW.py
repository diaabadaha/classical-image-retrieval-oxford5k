from pathlib import Path
import numpy as np
from tqdm import tqdm
import helpers.ioStore as ioStore


def buildHistogram(wordIds, vocabSize):
    if wordIds.shape[0] == 0:
        return np.zeros((vocabSize,), dtype=np.float32)
    return np.bincount(wordIds, minlength=vocabSize).astype(np.float32)


def computeIdf(histograms):
    N = histograms.shape[0]
    df = np.count_nonzero(histograms > 0, axis=0).astype(np.float32)
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
    return idf.astype(np.float32)


def nearestCenterIds(desc, centers):
    if desc.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    chunkSize = 2048
    ids = np.empty((desc.shape[0],), dtype=np.int32)

    centersT = centers.T
    centersNorm2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1, K)

    for start in range(0, desc.shape[0], chunkSize):
        end = min(start + chunkSize, desc.shape[0])
        d = desc[start:end]  # (c, 128)
        dNorm2 = np.sum(d * d, axis=1, keepdims=True)  # (c, 1)
        dist2 = dNorm2 + centersNorm2 - 2.0 * (d @ centersT)  # (c, K)
        ids[start:end] = np.argmin(dist2, axis=1).astype(np.int32)

    return ids


def encodeAllImages():
    descriptorsDir = Path("dataset/descriptors")
    outputDir = Path("dataset/vbow")
    ioStore.ensureDir(outputDir)

    centers = ioStore.loadCodebook(Path("dataset/codebook/codebook.npy"))
    vocabSize = centers.shape[0]

    descriptorPaths = ioStore.listAllDescriptorFiles(descriptorsDir)
    if len(descriptorPaths) == 0:
        raise RuntimeError(f"No descriptor files found in: {descriptorsDir}")

    rawHistograms = []
    imageNames = []

    for p in tqdm(descriptorPaths, desc="Building raw BoVW histograms"):
        desc = np.load(str(p)).astype(np.float32) 
        if desc.ndim != 2 or desc.shape[1] != 128:
            desc = np.zeros((0, 128), dtype=np.float32)

        wordIds = nearestCenterIds(desc, centers)
        hist = buildHistogram(wordIds, vocabSize)
        rawHistograms.append(hist)
        imageNames.append(p.stem + ".jpg")

    rawHistograms = np.vstack(rawHistograms).astype(np.float32)  # (N, K)

    idf = computeIdf(rawHistograms)
    ioStore.saveNpy(outputDir / "idf.npy", idf)

    tfidf = rawHistograms * idf[None, :]
    tfidf = np.vstack(
        [ioStore.l2Normalize(v) for v in tqdm(tfidf, desc="Normalizing TF-IDF vectors")]
    )

    ioStore.saveNpy(outputDir / "vbowTfidf.npy", tfidf.astype(np.float32))
    ioStore.saveJson(outputDir / "imageNames.json", imageNames)

    print(f"Saved TF-IDF matrix: {outputDir / 'vbowTfidf.npy'}  shape={tfidf.shape}")
    print(f"Saved IDF: {outputDir / 'idf.npy'}  shape={idf.shape}")
    print(
        f"Saved image names: {outputDir / 'imageNames.json'}  count={len(imageNames)}"
    )


if __name__ == "__main__":
    encodeAllImages()
