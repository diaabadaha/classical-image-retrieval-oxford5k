import numpy as np
import helpers.ioStore as ioStore


def nearestCenterIds(desc, centers):
    if desc.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    chunkSize = 2048
    ids = np.empty((desc.shape[0],), dtype=np.int32)

    centersT = centers.T
    centersNorm2 = np.sum(centers * centers, axis=1, keepdims=True).T

    for start in range(0, desc.shape[0], chunkSize):
        end = min(start + chunkSize, desc.shape[0])
        d = desc[start:end]

        dNorm2 = np.sum(d * d, axis=1, keepdims=True)
        dist2 = dNorm2 + centersNorm2 - 2.0 * (d @ centersT)
        ids[start:end] = np.argmin(dist2, axis=1).astype(np.int32)

    return ids


def buildHistogram(wordIds, vocabSize):
    if wordIds.shape[0] == 0:
        return np.zeros((vocabSize,), dtype=np.float32)
    return np.bincount(wordIds, minlength=vocabSize).astype(np.float32)


def encodeQuery(imageName):
    centers = ioStore.loadCodebook()
    idf = ioStore.loadIdf()

    if idf.shape[0] != centers.shape[0]:
        raise RuntimeError("IDF vocab size does not match codebook size")

    desc = ioStore.loadDescriptorsForImage(imageName)
    wordIds = nearestCenterIds(desc, centers)
    hist = buildHistogram(wordIds, centers.shape[0])

    tfidf = hist * idf
    queryVec = ioStore.l2Normalize(tfidf)
    return queryVec
