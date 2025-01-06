from dataset import my_dataset
from config.get_config import get_merge_config
from torch.utils.data import DataLoader

config = get_merge_config()
train_image_path = config.get("train_image_path")
train_label_path = config.get("train_label_path")
test_image_path = config.get("test_image_path")
test_label_path = config.get("test_label_path")

train_dataset = my_dataset(train_image_path, train_label_path)
test_dataset = my_dataset(train_image_path, train_label_path)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.get("batch_size"),
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=config.get("batch_size"),
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
