from tqdm import tqdm

for number in tqdm(range(100_000)):
    number ** number
