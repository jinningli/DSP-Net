from dataset.tripleDataset import TripleDataset

def CreateDataset(opt):
    # only support aligned
    dataset = TripleDataset()
    print("Aligned dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset