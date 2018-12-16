import sys


VOCAB = {}
VOCAB_INVERSE = {}


def instruction_to_string(instruction, config):

    if sys.version_info[0] == 2:
        if len(VOCAB) == 0:
            vocab_path = config["vocab_file"]
            with open(vocab_path) as f:
                for line in f.xreadlines():
                    token = line.strip().decode("utf-8")
                    VOCAB[token] = len(VOCAB)
            for token, i in VOCAB.iteritems():
                VOCAB_INVERSE[i] = token
    elif sys.version_info[0] == 3:
        # python 3 handling
        if len(VOCAB) == 0:
            vocab_path = config["vocab_file"]
            with open(vocab_path) as f:
                for line in f.readlines():
                    token = line.strip()
                    VOCAB[token] = len(VOCAB)
            for token, i in VOCAB.items():
                VOCAB_INVERSE[i] = token
            VOCAB_INVERSE[len(VOCAB)] = "$UNK$"
    else:
        raise AssertionError("Unknown version. Found " + str(sys.version_info[0]) + " expected python 2 or 3")

    return " ".join([VOCAB_INVERSE[i] for i in instruction])
