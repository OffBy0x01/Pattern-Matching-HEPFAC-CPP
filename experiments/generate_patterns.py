#!/usr/bin/env python3

import argparse
import string
import random

def str_generator(size=4, chars=string.ascii_letters + string.digits):
    return (''.join(random.choice(chars) for _ in range(size))) + '\n'

# Arg Parsing
parser = argparse.ArgumentParser()
group = parser.add_argument_group("Required Arguments")
group.add_argument("-l", "--patternlength", nargs='?', default=5, required=True)
group.add_argument("-c", "--count",  nargs='?', default=4, required=True)
args = parser.parse_known_args()[0]

with open("len%s_count%s.txt" % (args.patternlength, args.count), "w") as outfile:
    for line in range(int(args.count)):
        outfile.write(str_generator(size=int(args.patternlength)))
