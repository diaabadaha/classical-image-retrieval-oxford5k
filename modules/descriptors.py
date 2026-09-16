from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import helpers.ioStore as ioStore

def rootSift(descriptors):
    if descriptors is None or len(descriptors) == 0:
        return descriptors
    eps = 1e-12
    l1 = descriptors.sum(axis=1, keepdims=True) + eps
    descriptors = descriptors / l1
    descriptors = np.sqrt(descriptors)
    return descriptors

def extractSiftFromImage(grayImage, sift):
    keyPoints, descriptors = sift.detectAndCompute(grayImage, None)
    if descriptors is None:
        return [], None
    descriptors = descriptors.astype(np.float32)
    descriptors = rootSift(descriptors)
    return keyPoints, descriptors

def saveDescriptors(outputDir, imageName, descriptors):
    outPath = Path(outputDir) / (Path(imageName).stem + ".npy")
    if descriptors is None:
        np.save(str(outPath), np.zeros((0, 128), dtype=np.float32))
    else:
        np.save(str(outPath), descriptors)

def saveKeypoints(keypointsDir, imageName, keyPoints):
    kpPath = Path(keypointsDir) / (Path(imageName).stem + ".npy")
    if keyPoints is None or len(keyPoints) == 0:
        np.save(str(kpPath), np.zeros((0, 2), dtype=np.float32))
        return

    # Store only (x, y) for geometry
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keyPoints], dtype=np.float32)
    np.save(str(kpPath), pts)

def extractAllSift():
    imagesDir = Path("dataset/preprocessed")
    descriptorsDir = Path("dataset/descriptors")
    keypointsDir = Path("dataset/keypoints")

    ioStore.ensureDir(descriptorsDir)
    ioStore.ensureDir(keypointsDir)

    imagePaths = ioStore.listAllPreprocessedImages(imagesDir)
    if len(imagePaths) == 0:
        raise RuntimeError(f"No images found in: {imagesDir}")

    sift = cv2.SIFT_create()

    processedCount = 0
    failedCount = 0
    totalKeyPoints = 0

    for inputPath in tqdm(imagePaths, desc="Extracting SIFT"):
        grayImage = cv2.imread(str(inputPath), cv2.IMREAD_GRAYSCALE)
        if grayImage is None:
            failedCount += 1
            continue

        keyPoints, descriptors = extractSiftFromImage(grayImage, sift)
        saveDescriptors(descriptorsDir, inputPath.name, descriptors)
        saveKeypoints(keypointsDir, inputPath.name, keyPoints)

        totalKeyPoints += len(keyPoints)
        processedCount += 1

    avgKeyPoints = (totalKeyPoints / processedCount) if processedCount > 0 else 0.0
    print(f"Done. Total: {len(imagePaths)}, processed: {processedCount}, failed: {failedCount}")
    print(f"Descriptors folder: {descriptorsDir}")
    print(f"Keypoints folder:   {keypointsDir}")
    print(f"Average keyPoints per image: {avgKeyPoints:.2f}")

if __name__ == "__main__":
    extractAllSift()
