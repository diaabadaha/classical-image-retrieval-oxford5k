from pathlib import Path
import cv2
from tqdm import tqdm
import helpers.ioStore as ioStore

def resizeKeepAspect(grayImage):
    h, w = grayImage.shape[:2]
    m = max(h, w)

    if m <= 1024:
        return grayImage

    scale = 1024 / float(m)
    newW = int(round(w * scale))
    newH = int(round(h * scale))
    return cv2.resize(grayImage, (newW, newH), interpolation=cv2.INTER_AREA)

def preprocessImage(bgr):
    grayImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resizedGrayImage = resizeKeepAspect(grayImage)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    finalImage = clahe.apply(resizedGrayImage)
    return finalImage

def preprocessAllImages():
    imagesDir = Path("dataset/images")
    outputDir = Path("dataset/preprocessed")
    ioStore.ensureDir(outputDir)

    imagePaths = ioStore.listAllImages(imagesDir)
    if len(imagePaths) == 0:
        raise RuntimeError(f"No images found in: {imagesDir}")

    processedCount = 0
    failedCount = 0

    for inputPath in tqdm(imagePaths, desc="Preprocessing images"):
        bgr = cv2.imread(str(inputPath), cv2.IMREAD_COLOR)
        if bgr is None:
            failedCount += 1
            continue

        finalImage = preprocessImage(bgr)
        succeeded = cv2.imwrite(str(outputDir / inputPath.name), finalImage) 

        if succeeded:
            processedCount += 1
        else:
            failedCount += 1

    print(f"Done. Total: {len(imagePaths)}, processed: {processedCount}, failed: {failedCount}")
    print(f"Output folder: {outputDir}")

if __name__ == "__main__":
    preprocessAllImages()
