import sys
sys.path.insert(0, 'build')
import attention_cuda_py
print('Success!')
print('CUDA available:', attention_cuda_py.cuda_available())
