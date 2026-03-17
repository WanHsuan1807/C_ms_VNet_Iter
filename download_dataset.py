from tdsc_abus2023_pytorch import TDSC, DataSplits

dataset = TDSC(path="./data", split=DataSplits.TEST, download=True)

print("len(dataset) =", len(dataset))
volume, mask, label, bbx = dataset[0]

# 盡量印出你最需要的資訊
print("volume:", type(volume), getattr(volume, "shape", None), getattr(volume, "dtype", None))
print("mask:  ", type(mask), getattr(mask, "shape", None), getattr(mask, "dtype", None))
print("label:", label)
print("bbx:  ", bbx)
