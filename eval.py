import io
import sys
import json
import argparse
import signal
import contextlib
from tqdm import tqdm

@contextlib.contextmanager
def capture_output():
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    try:
        yield stdout_buf, stderr_buf
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

def timeout_handler(*_):
    raise Exception('code execution timed out')

def exec_with_timeout(code, timeout):
    signal.alarm(timeout)
    exec(code)
    signal.alarm(0)

def exec_tests(code, test_list, setup_code, timeout=10):
    res = {'test_results': [], 'code_error': None, 'pass': 0}

    # NOTE -- don't try this at home!
    with capture_output() as (stdout, stderr):
        signal.signal(signal.SIGALRM, timeout_handler)

        try:
            exec_with_timeout(code, timeout)
        except Exception as e:
            res['code_error'] = str(e)
            return res

        for test in test_list:
            try:
                exec_with_timeout(f'{code}\n{setup_code}\n{test}', timeout)
                res['test_results'].append(('pass', None))
                res['pass'] += 1
            except Exception as e:
                res['test_results'].append(('fail', str(e)))

        res['stdout'] = stdout.getvalue()
        res['stderr'] = stderr.getvalue()

    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--timeout', default=10, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.input, 'r') as f_in, open(args.output, 'w') as f_out:
        for line in tqdm(f_in):
            obj = json.loads(line)
            test_list = obj['item']['test_list']
            setup_code = obj['item']['setup_code']
            obj['results'] = [exec_tests(c, test_list, setup_code, args.timeout) for c in obj['resps']]
            json.dump(obj, f_out)
            f_out.write('\n')

if __name__ == '__main__':
    main()
