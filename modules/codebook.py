from pathlib import Path
import numpy as np
from tqdm import tqdm
import helpers.ioStore as ioStore
from sklearn.cluster import MiniBatchKMeans # type: ignore


def loadDescriptors(npyPath):
    desc = np.load(str(npyPath))
    if desc is None or desc.ndim != 2 or desc.shape[1] != 128:
        return np.zeros((0, 128), dtype=np.float32)
    return desc.astype(np.float32)


def sampleDescriptors(descriptorPaths):
    maxDescriptors = 300000
    perImageCap = 300
    rng = np.random.default_rng(42)

    sampledList = []
    total = 0

    for p in descriptorPaths:
        desc = loadDescriptors(p)
        n = desc.shape[0]
        if n == 0:
            continue

        take = min(perImageCap, n, maxDescriptors - total)
        if take <= 0:
            break

        idx = rng.choice(n, size=take, replace=False)
        sampledList.append(desc[idx])
        total += take

        if total >= maxDescriptors:
            break

    if len(sampledList) == 0:
        raise RuntimeError(
            "No descriptors found to sample (are your .npy files empty?)"
        )

    return np.vstack(sampledList).astype(np.float32)


def buildCodeBookKmeans(sampledDescriptors):
    clusterCount = 2000
    batchSize = 10000

    kmeans = MiniBatchKMeans(
        n_clusters=clusterCount,
        batch_size=batchSize,
        random_state=42,
        verbose=0,
        n_init="auto",
    )

    kmeans.fit(sampledDescriptors)
    centers = kmeans.cluster_centers_.astype(np.float32)
    return centers


def buildCodeBook():
    descriptorsDir = Path("dataset/descriptors")
    outputDir = Path("dataset/codeBook")
    ioStore.ensureDir(outputDir)

    descriptorPaths = ioStore.listAllDescriptorFiles(descriptorsDir)
    if len(descriptorPaths) == 0:
        raise RuntimeError(f"No descriptor .npy files found in: {descriptorsDir}")

    sampledDescriptors = sampleDescriptors(descriptorPaths)
    print(f"Sampled descriptors: {sampledDescriptors.shape}")

    centers = buildCodeBookKmeans(sampledDescriptors)
    ioStore.saveNpy(outputDir / "codeBook.npy", centers)
    print(f"Saved codeBook: {outputDir / 'codeBook.npy'}  shape={centers.shape}")


if __name__ == "__main__":
    buildCodeBook()
