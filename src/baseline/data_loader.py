from torch.utils.data import DataLoader,Dataset
from src.baseline.vocabulary import Vocabulary
from src.baseline.coa_dataset import CoADataset
from src.baseline.caps_collate import CapsCollate


def get_loader(root_folder, annotation_file, transform, 
               batch_size=32, num_workers=2, shuffle=True, pin_memory=True, vocab=None):
    
    dataset = CoADataset(root_folder, annotation_file, transform=transform, vocab=vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapsCollate(pad_idx=pad_idx)
    )

    return loader, dataset

def get_loaders(root_folder, train_annotation_file, val_annotation_file, test_annotation_file, 
                transform, batch_size=32, num_workers=2, shuffle=True, pin_memory=True, vocab=None):
    
    train_loader, train_dataset = get_loader(root_folder, train_annotation_file, transform, 
                                             batch_size, num_workers,shuffle, pin_memory, vocab)
    val_loader, val_dataset = get_loader(root_folder, val_annotation_file, transform, 
                                         batch_size, num_workers,shuffle, pin_memory, vocab)
    test_loader, test_dataset = get_loader(root_folder, test_annotation_file, transform, 
                                           batch_size, num_workers,shuffle, pin_memory, vocab)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
