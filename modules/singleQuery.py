import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import helpers.ioStore as ioStore
import encodeQuery
import helpers.pcaProject as pcaProject


def findLandmarkForQuery(groundTruth, queryName):
    for landmarkName, data in groundTruth.items():
        if queryName in data.get("query", []):
            return landmarkName
    return None


def getLabelForImage(name, goodSet, okSet, junkSet):
    if name in goodSet:
        return "good"
    if name in okSet:
        return "ok"
    if name in junkSet:
        return "junk"
    return "bad"


def getBorderStyle(label):
    if label == "good":
        return ("green", 4)
    if label == "ok":
        return ("blue", 4)
    if label == "junk":
        return ("gold", 4)
    return ("red", 4)


def loadRgbImage(imagesDir, imageName):
    imgPath = Path(imagesDir) / imageName
    img = cv2.imread(str(imgPath), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def retrieveTopKFullScan(queryVec, databaseVectors, imageNames, excludeNamesSet, topK):
    scores = databaseVectors @ queryVec
    rankedIdx = np.argsort(-scores)

    results = []
    for idx in rankedIdx:
        name = imageNames[idx]
        if name in excludeNamesSet:
            continue
        results.append((name, float(scores[idx])))
        if len(results) >= topK:
            break
    return results


def precisionAtK(rankedNames, relevantSet, k):
    if k <= 0:
        return 0.0
    top = rankedNames[:k]
    hits = sum(1 for n in top if n in relevantSet)
    return hits / float(k)


def recallAtK(rankedNames, relevantSet, k):
    if len(relevantSet) == 0 or k <= 0:
        return 0.0
    top = rankedNames[:k]
    hits = sum(1 for n in top if n in relevantSet)
    return hits / float(len(relevantSet))


def computeAp(rankedNames, relevantSet):
    if len(relevantSet) == 0:
        return 0.0

    hitCount = 0
    precisionSum = 0.0

    for rank, name in enumerate(rankedNames, start=1):
        if name in relevantSet:
            hitCount += 1
            precisionSum += hitCount / rank

    return precisionSum / len(relevantSet)


def demoSingleQuery():
    queryName = "all_souls_000013.jpg"
    showTopK = 10

    usePca = False
    pcaDim = 256

    imagesDir = Path("dataset/images")

    groundTruth = ioStore.loadGroundTruth()
    landmarkName = findLandmarkForQuery(groundTruth, queryName)
    if landmarkName is None:
        raise RuntimeError(f"Query image not found in ground truth: {queryName}")

    data = groundTruth[landmarkName]
    goodSet = set(data.get("good", []))
    okSet = set(data.get("ok", []))
    junkSet = set(data.get("junk", []))
    relevantSet = goodSet.union(okSet)

    databaseVectors = ioStore.loadDatabaseVectors()
    imageNames = ioStore.loadImageNames()

    queryVec = encodeQuery.encodeQuery(queryName)

    if usePca:
        mean, components = pcaProject.loadPcaModel(pcaDim)
        queryVec = pcaProject.applyPcaToVector(queryVec, mean, components)
        databaseVectors = pcaProject.loadPcaDatabaseVectors(pcaDim)

    excludeNamesSet = set(junkSet)
    excludeNamesSet.add(queryName)

    results = retrieveTopKFullScan(
        queryVec, databaseVectors, imageNames, excludeNamesSet, topK=2000
    )

    rankedNames = [name for (name, score) in results]

    p5 = precisionAtK(rankedNames, relevantSet, 5)
    r5 = recallAtK(rankedNames, relevantSet, 5)
    p10 = precisionAtK(rankedNames, relevantSet, 10)
    r10 = recallAtK(rankedNames, relevantSet, 10)
    ap = computeAp(rankedNames, relevantSet)

    print("========================================")
    print("Single Query Demo")
    print("========================================")
    print(f"Query: {queryName}  landmark={landmarkName}")
    print(f"usePca={usePca} pcaDim={pcaDim}")
    print(f"AP: {ap:.4f}")
    print(f"Precision@5:  {p5:.4f}   Recall@5:  {r5:.4f}")
    print(f"Precision@10: {p10:.4f}   Recall@10: {r10:.4f}")
    print("========================================")

    queryImg = loadRgbImage(imagesDir, queryName)
    if queryImg is None:
        raise RuntimeError(f"Could not load query image: {imagesDir / queryName}")

    shown = rankedNames[:showTopK]
    cols = showTopK + 1
    plt.figure(figsize=(3.2 * cols, 4.2))

    ax = plt.subplot(1, cols, 1)
    ax.imshow(queryImg)
    ax.set_title(f"QUERY\n{landmarkName}", fontsize=10)
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=4,
        )
    )

    for i, name in enumerate(shown, start=1):
        img = loadRgbImage(imagesDir, name)
        if img is None:
            continue

        label = getLabelForImage(name, goodSet, okSet, junkSet)
        borderColor, borderWidth = getBorderStyle(label)

        ax = plt.subplot(1, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"#{i}  {label}", fontsize=9)
        ax.axis("off")
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                edgecolor=borderColor,
                linewidth=borderWidth,
            )
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demoSingleQuery()
