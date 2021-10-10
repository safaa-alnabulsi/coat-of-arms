from torch.utils.data import DataLoader,Dataset
from src.baseline.vocabulary import Vocabulary
from src.baseline.coa_dataset import CoADataset
from src.baseline.caps_collate import CapsCollate

def get_loader(root_folder, train_annotation_file, test_annotation_file, transform, batch_size=32, num_workers=8, shuffle=True, pin_memory=True):
    
    train_dataset = CoADataset(root_folder, train_annotation_file, transform=transform)
    test_dataset = CoADataset(root_folder, test_annotation_file, transform=transform)
    pad_idx = train_dataset.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapsCollate(pad_idx=pad_idx),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapsCollate(pad_idx=pad_idx),
    )
    
    return train_loader, test_loader, train_dataset, test_dataset
