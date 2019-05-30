from dataloader.customDataLoader import CustomDataLoader

def CreateDataLoader(opt):
    data_loader = CustomDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


