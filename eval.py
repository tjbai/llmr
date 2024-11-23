import json
import argparse

def eval_tests(code, tests):
    pass_ct = 0
    for test in tests:
        try:
            eval(f'{code}\n\n{tests[0]}')
            pass_ct += 1
        except Exception:
            continue
    return pass_ct

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.input, 'r') as in_f, open(args.output, 'r') as out_f:
        for line in in_f:
            pass

if __name__ == '__main__':
    main()
