with open('train_vids.txt', 'r') as f:
    vids = [v.strip() for v in f.readlines()]


vids1, vids2 = [], []
for i, vid in enumerate(vids):
    if i < 3500:
        vids1.append(vid)
    else:
        vids2.append(vid)


with open('train_vids1.txt', 'w') as f:
    for vid in vids1:
        f.writelines(vid+'\n')
with open('train_vids2.txt', 'w') as f:
    for vid in vids2:
        f.writelines(vid+'\n')
