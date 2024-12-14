import json
import ast
import argparse
from pathlib import Path
import tempfile
import subprocess
from tqdm import tqdm

RESULT_DELIM = 'RESULT_JSON: '

def create_test_file(code, test_list, setup_code):
    template = f'''
import unittest
import signal
import sys
from typing import *

{code}

{setup_code or ''}

class TestCode(unittest.TestCase):
    def setUp(self):
        signal.alarm(10)
        
    def tearDown(self):
        signal.alarm(0)
        
    def run(self, result=None):
        try:
            return super().run(result)
        except Exception as e:
            self.fail(str(e))
'''

    for i, test in enumerate(test_list):
        template += f'''
    def test_{i + 1}(self):
        {test}
'''

    template += f'''
if __name__ == '__main__':
    runner = unittest.TextTestRunner(stream=sys.stdout)
    result = runner.run(unittest.makeSuite(TestCode))
    
    import json
    print('{RESULT_DELIM}', json.dumps({{
        'tests_run': result.testsRun,
        'tests_passed': result.testsRun - len(result.failures) - len(result.errors),
        'failures': [str(fail[1]) for fail in result.failures],
        'errors': [str(error[1]) for error in result.errors]
    }}))
'''

    return template

def error_all(test_list, error_msg, stdout='', stderr=''):
    return {
        'test_results': [('fail', error_msg) for _ in test_list],
        'code_error': error_msg,
        'pass': 0,
        'stdout': stdout,
        'stderr': stderr
    }

def parse_test_results(stdout, process_output, test_list):
    result_line = next((line for line in stdout if line.startswith(RESULT_DELIM)), None)
    if not result_line:
        return error_all(test_list, 'could not parse', process_output.stdout, process_output.stderr)
    result = json.loads(result_line.replace(RESULT_DELIM, ''))
    
    test_results = []
    for i in range(result['tests_run']):
        failure_msg = None
        if i < len(result['failures']):
            failure_msg = result['failures'][i]
        elif i < len(result['errors']):
            failure_msg = result['errors'][i]
        test_results.append(('pass' if not failure_msg else 'fail', failure_msg))
    
    return {
        'test_results': test_results,
        'code_error': None,
        'pass': result['tests_passed'],
        'stdout': process_output.stdout,
        'stderr': process_output.stderr
    }

def run_tests(code, test_list, setup_code, timeout):
    try:
        ast.parse(code)
    except SyntaxError as e:
        return error_all(test_list, f'Syntax error: {str(e)}', stderr=str(e))

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / 'test_code.py'
        with open(test_file, 'w') as f:
            f.write(create_test_file(code, test_list, setup_code))
        
        try:
            process = subprocess.run(['python3', str(test_file)], capture_output=True, text=True, timeout=timeout)
            return parse_test_results(process.stdout.split('\n'), process, test_list)
        except subprocess.TimeoutExpired:
            return error_all(test_list, 'execution timed out')
        except Exception as e:
            return error_all(test_list, str(e))

def parse_code(text):
    return text.split('```')[0].strip()

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

            code = [parse_code(c) for c in obj['text']]
            test_list = obj['item']['test_list']
            setup_code = obj['item'].get('test_setup_code', '')

            obj['results'] = [run_tests(c, test_list, setup_code, args.timeout) for c in code]
            json.dump(obj, f_out)
            f_out.write('\n')

if __name__ == '__main__':
    main()