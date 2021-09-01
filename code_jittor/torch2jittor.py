import os
import glob
from jittor.utils.pytorch_converter import convert

path = './code'
filenames = glob.glob(os.path.join(path, '*', '*.py'))
for filename in filenames:
    with open(filename) as f:
        
        pytorch_code = f.read()
        try:
            jittor_code = convert(pytorch_code)
        except Exception as e:
            print(f'{filename}: {e}')
        jittor_filename = filename.replace('./code', './code_jittor')
        dir_name = os.path.dirname(jittor_filename)
        if(not os.path.exists(dir_name)):
            os.makedirs(dir_name)
            with open(jittor_filename, 'w') as f1:
                f1.write(jittor_code)
        