import argparse

parser = argparse.ArgumentParser(description='Translate with a trained model.')
parser.add_argument('--align', type=str,
                    help='location of the alignment data')
parser.add_argument('--src', type=str,
                    help='location of the src data')
parser.add_argument('--trg', type=str,
                    help='location of the trg data')
parser.add_argument('--left_pad', type=int, default=3,
                    help='Number of padding symbols to add to the beginning of the trg sentence.')
parser.add_argument('--directory', type=str,
                    help='where to put the files')
parser.add_argument('--num_valid', type=int, default=3000)
args = parser.parse_args()

left_pad = args.left_pad

TRG_EPSILON = '@@@@'
SRC_EPSILON = '@@@'
START_PAD = '@str@@'


def readf(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


align = readf(args.align)

unaligned = []


def parse_alignment(i, line):
    line = line.split(' ')
    pairs = []
    for pair in line:
        split_p = pair.split('-')
        if split_p[0] == '' or split_p[1] == '':
            unaligned.append(i)
            return None
        pairs.append((int(split_p[0]), int(split_p[1])))
    return pairs


print("Processing... This could take a minute.")


def remove_lines_from_file(file, lines_to_remove):
    dictionary = {a: 1 for a in lines_to_remove}
    f = open(file, "r+")
    d = f.readlines()
    f.seek(0)
    for count, str in enumerate(d):
        if dictionary.get(count, 0) != 1:
            f.write(str)
    f.truncate()
    f.close()


parsedLines = [parse_alignment(i, l) for i, l in enumerate(align)]

if len(unaligned) > 0:
    print(len(unaligned))
    remove_lines_from_file(args.src, unaligned)
    remove_lines_from_file(args.trg, unaligned)
    remove_lines_from_file(args.align, unaligned)
    print('Removing ' + str(len(
        unaligned)) + ' unaligned lines from src / trg / alignment files.\n\nTo continue, please rerun this command.')
    import sys

    sys.exit()


src = open(args.src, 'r')
trg = open(args.trg, 'r')

directory = args.directory
num_valid = args.num_valid

src_out_val = open(directory + "valid_src.txt", 'w')
src_out_train =  open(directory + "train_src.txt", 'w')
trg_out_val =  open(directory + "valid_trg.txt", 'w')
trg_out_train =  open(directory + "train_trg.txt", 'w')


def read_single_line(input_line):
    return input_line.strip().split()

print(len(parsedLines))
for a in range(len(parsedLines)):
    l = parsedLines[a]
    assert l != None
    src_split = read_single_line(src.readline())
    trg_split = read_single_line(trg.readline())
    lenS, lenT = len(src_split), len(trg_split)
    cost = 0

    for i in range(args.left_pad):
        trg_split.insert(0, START_PAD)
        cost += 1

    for pair in l:
        i, j = pair
        if i > j:
            diff = i - j
            if diff > cost:
                local_cost = diff - cost

                for d in range(local_cost):
                    trg_split.insert(j + cost, SRC_EPSILON)
                    cost += 1

    # pad at the end,
    # so that both src and trg sequences are of the same size.
    if lenS > lenT + cost:
        trg_split.extend([SRC_EPSILON] * (lenS - lenT - cost))
    elif lenT + cost > lenS:
        src_split.extend([TRG_EPSILON] * (lenT + cost - lenS))

    if a<num_valid:
        src_out_val.write(" ".join(src_split) + '\n')
        trg_out_val.write(" ".join(trg_split) + '\n')
    else:
        src_out_train.write(" ".join(src_split) + '\n')
        trg_out_train.write(" ".join(trg_split) + '\n')












