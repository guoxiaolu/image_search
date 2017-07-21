
# probability threshold of label
TOP_TH = 0.1
# label of other
LABEL_OTHER = 10000

def read_labels(fname='./label.txt'):
    labels_name = {}
    labels_id = {}
    f = open(fname, mode='r')
    for line in f.readlines():
        con = line.strip().split(' ')
        name = con[-1]
        id = int(con[0])
        labels_id[id] = name
        if name not in labels_name.keys():
            labels_name[name] = id
    f.close()
    return labels_id, labels_name

def get_labels(labels_id, labels_name, label, prob):
    label_filter = []
    top_prob = prob[0]
    if top_prob != top_prob or top_prob < TOP_TH:
        label_filter = [LABEL_OTHER, LABEL_OTHER, LABEL_OTHER]
    else:
        for id in label:
            label_filter.append(labels_name[labels_id[id]])
    return label_filter