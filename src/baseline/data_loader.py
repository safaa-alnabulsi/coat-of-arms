from torch.utils.data import DataLoader,Dataset
from src.baseline.vocabulary import Vocabulary
from src.baseline.coa_dataset import CoADataset
from src.baseline.caps_collate import CapsCollate

def get_loader(root_folder, train_annotation_file, val_annotation_file, test_annotation_file, transform, batch_size=32, num_workers=2, shuffle=True, pin_memory=True, vocab=None):
    
    train_dataset = CoADataset(root_folder, train_annotation_file, transform=transform, vocab=vocab)
    val_dataset = CoADataset(root_folder, val_annotation_file, transform=transform, vocab=vocab)
    test_dataset = CoADataset(root_folder, test_annotation_file, transform=transform, vocab=vocab)
    pad_idx = train_dataset.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapsCollate(pad_idx=pad_idx)
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapsCollate(pad_idx=pad_idx)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CapsCollate(pad_idx=pad_idx)
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
