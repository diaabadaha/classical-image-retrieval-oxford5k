import time
import numpy as np
import helpers.ioStore as ioStore
import encodeQuery
import helpers.pcaProject as pcaProject


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


def precisionAtK(rankedNames, relevantSet, k):
    top = rankedNames[:k]
    hits = sum(1 for n in top if n in relevantSet)
    return hits / float(k)


def recallAtK(rankedNames, relevantSet, k):
    if len(relevantSet) == 0:
        return 0.0
    top = rankedNames[:k]
    hits = sum(1 for n in top if n in relevantSet)
    return hits / float(len(relevantSet))


def evaluate(usePca, pcaDim):
    groundTruth = ioStore.loadGroundTruth()
    if not groundTruth:
        raise RuntimeError("groundtruth.json not found or empty")

    imageNames = ioStore.loadImageNames()

    if usePca:
        databaseVectors = pcaProject.loadPcaDatabaseVectors(pcaDim)
        mean, components = pcaProject.loadPcaModel(pcaDim)
    else:
        databaseVectors = ioStore.loadDatabaseVectors()
        mean, components = None, None

    if databaseVectors.shape[0] != len(imageNames):
        raise RuntimeError("Database vectors rows do not match imageNames length")

    kList = [5, 10]
    apList = []
    pAtK = {k: [] for k in kList}
    rAtK = {k: [] for k in kList}
    queryTimesMs = []

    totalQueries = 0

    for _, data in groundTruth.items():
        goodSet = set(data.get("good", []))
        okSet = set(data.get("ok", []))
        junkSet = set(data.get("junk", []))
        queryList = data.get("query", [])

        relevantSetBase = goodSet.union(okSet)

        for queryName in queryList:
            totalQueries += 1

            relevantSet = set(relevantSetBase)
            relevantSet.discard(queryName)

            excludeNamesSet = set(junkSet)
            excludeNamesSet.add(queryName)

            t0 = time.perf_counter()

            queryVec = encodeQuery.encodeQuery(queryName)

            if usePca:
                queryVec = pcaProject.applyPcaToVector(queryVec, mean, components)

            results = retrieveTopKFullScan(
                queryVec=queryVec,
                databaseVectors=databaseVectors,
                imageNames=imageNames,
                excludeNamesSet=excludeNamesSet,
                topK=2000,
            )

            rankedNames = [name for (name, score) in results]

            t1 = time.perf_counter()
            queryTimesMs.append((t1 - t0) * 1000.0)

            apList.append(computeAp(rankedNames, relevantSet))
            for k in kList:
                pAtK[k].append(precisionAtK(rankedNames, relevantSet, k))
                rAtK[k].append(recallAtK(rankedNames, relevantSet, k))

    mAp = float(np.mean(apList)) if apList else 0.0
    avgTime = float(np.mean(queryTimesMs)) if queryTimesMs else 0.0

    print("========================================")
    print("Oxford5k Retrieval Evaluation")
    print("========================================")
    print(f"usePca={usePca}  pcaDim={pcaDim}")
    print(f"Queries evaluated: {totalQueries}")
    print(f"mAP: {mAp:.4f}")
    for k in kList:
        print(f"Precision@{k}: {float(np.mean(pAtK[k])):.4f}")
        print(f"Recall@{k}:    {float(np.mean(rAtK[k])):.4f}")
    print(f"Avg retrieval time per query: {avgTime:.2f} ms")
    print("========================================")


if __name__ == "__main__":
    evaluate(usePca=True, pcaDim=256)
