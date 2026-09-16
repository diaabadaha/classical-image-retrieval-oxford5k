import helpers.ioStore as ioStore
import helpers.pcaProject as pcaProject


def buildPca():
    databaseVectors = ioStore.loadDatabaseVectors()
    pcaProject.fitAndSavePca(databaseVectors, 256)


if __name__ == "__main__":
    buildPca()
