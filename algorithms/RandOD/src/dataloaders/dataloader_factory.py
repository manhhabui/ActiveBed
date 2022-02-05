from algorithms.RandOD.src.dataloaders.VOC_Dataloader import PascalVOCDataset

train_dataloaders_map = {
    "PascalVOC": PascalVOCDataset
}

test_dataloaders_map = {
    "PascalVOC": PascalVOCDataset
}

def get_train_dataloader(name):
    if name not in train_dataloaders_map:
        raise ValueError("Name of train dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return train_dataloaders_map[name](**kwargs)

    return get_dataloader_fn


def get_test_dataloader(name):
    if name not in test_dataloaders_map:
        raise ValueError("Name of test dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return test_dataloaders_map[name](**kwargs)

    return get_dataloader_fn