import csv

# import nltk

LABEL_TO_INDEX = {
    'business':                  0,
    'computers':                 1,
    'culture-arts-entertainment':2,
    'education-science':         3,
    'engineering':               4,
    'health':                    5,
    'politics-society':          6,
    'sports':                    7
}

def create_tsv_file(path_in, path_out):

    with open(path_in,'r') as f, open(path_out,'w') as fw:
        writer = csv.writer(fw, delimiter='\t')
        writer.writerow(['label','body'])
        for line in f:
            tokens = [x.lower() for x in line.split()]
            label = LABEL_TO_INDEX[tokens[-1]]
            body = ' '.join(tokens[:-1])
            writer.writerow([label, body])


def _tokenize(text):
    # return [x.lower() for x in nltk.word_tokenize(text)]
    return [ x.lower() for x in text.split() ]


''' from https://github.com/pytorch/examples/blob/master/imagenet/main.py'''
class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 8 epochs"""
    lr = lr * (0.1 ** (epoch // 8))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def make_data(path):
    data_file = open(path, "r", encoding="utf8")
    train_file = open("data/aminer_train.tsv", "w", encoding="utf8")
    test_file = open("data/aminer_test.tsv", "w", encoding="utf8")

    lines = data_file.readlines()
    num = len(lines)
    print(num)

    # with open(path_in,'r') as f, open(path_out,'w') as fw:
    #     writer = csv.writer(fw, delimiter='\t')
    #     writer.writerow(['label','body'])
    #     for line in f:
    #         tokens = [x.lower() for x in line.split()]
    #         label = LABEL_TO_INDEX[tokens[-1]]
    #         body = ' '.join(tokens[:-1])
    #         writer.writerow([label, body])

    file = train_file
    print('Writing in train_file...')
    file.write('label\tbody\n')
    for i, line in enumerate(lines):
        # train: test = 4: 1
        word = line.split()
        tag = -1
        if word[0] == '搜学者':
            tag = 0
        elif word[0] == '搜文章':
            tag = 1
        elif word[0] == '搜会议':
            tag = 2
        sentence = ''
        for j in range(1, len(word)):
            sentence += word[j] + ' '
        file.write(str(tag) + '\t' + sentence + '\n')
        if file == train_file and i > num * 0.8:
            file = test_file
            print('Writing in dev_file...')
            file.write('label\tbody\n')

    data_file.close()
    train_file.close()
    test_file.close()

    print("done")

if __name__ == '__main__':
    make_data('data/data.txt')
