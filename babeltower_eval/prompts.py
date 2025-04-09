CUDA_CPP_TRANSLATE_ZERO_SHOT = """You are an expert in translating {obj.source} programs to {obj.target} programs. Given the {obj.source} program below, translate it to {obj.target}. Ensure that the {obj.target} program is exactly the same as the {obj.source} program input and output, and that the semantics of the original code are preserved.
Just generate the {obj.target} program and remove any unnecessary comments. Surround the generated {obj.target} program in [{obj.target}] and [/{obj.target}].  
### {obj.source} Program:
[{obj.source}]
{content}
[/{obj.source}]
### {obj.target} Version:
"""

CUDA_CPP_TRANSLATE_SYSTEM = """You are an expert in translating {obj.source} programs to {obj.target} programs. Given the {obj.source} program by User, translate it to {obj.target}. Ensure that the {obj.target} program is exactly the same as the {obj.source} program input and output, and that the semantics of the original code are preserved.
Just generate the {obj.target} program and remove any unnecessary comments. Surround the generated {obj.target} program in [{obj.target}] and [/{obj.target}].
"""

CUDA_CPP_TRANSLATE_USER = """
### {obj.source} Program:
[{obj.source}]
{content}
[/{obj.source}]
### {obj.target} Version:
"""

CUDA_CPP_TRANSLATE_TRAIN_SYSTEM = """You are an expert in translating {obj.source} programs to {obj.target} programs. Given the {obj.source} program by User, translate it to {obj.target}. Ensure that the {obj.target} program is exactly the same as the {obj.source} program input and output, and that the semantics of the original code are preserved.
Just generate the {obj.target} program and remove any unnecessary comments.
"""


CPP2CUDA_TRANSLATION_PROMPT = """
Please help me convert this CPU code into equivalent CUDA kernel code. The converted code should:

1. Preserve the original functionality
2. Process data elements in the same order
3. Keep the same input parameters and data handling logic
4. The generated code must be in the  [CODE] and [/CODE] tags.


Here is an example for you:

Cpp Code:
```cpp
void add_100(int numElements, int *data) {{
    for (int idx = 0; idx < numElements; idx++) {{
        data[idx] += 100;
    }}
}}
```

Cuda Code:
[CODE]
```cuda
__global__ void add_100(int numElements, int *data) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {{
        data[idx] += 100;
    }}
}}
```
[/CODE] 

Your task is to write a equivalent cuda kernel function for the following cpp function:

Cpp Code:
```cpp
{cpp_code}
```

Cuda Code:
"""

CUDA2CPP_TRANSLATION_PROMPT = """
Please help me convert this CUDA kernel code into equivalent CPU code. The converted code should:

1. Preserve the original functionality 
2. Process data elements in the same order
3. Keep the same input parameters and data handling logic
4. The generated code must be in the [CODE] and [/CODE] tags.

Here is an example for you:

Cuda Code:
```cuda
__global__ void add_100(int numElements, int *data) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {{
        data[idx] += 100;
    }}
}}
```

Cpp Code:
[CODE]
```cpp
void add_100(int numElements, int *data) {{
    for (int idx = 0; idx < numElements; idx++) {{
        data[idx] += 100;
    }}
}}
```
[/CODE]

Your task is to write an equivalent CPU function for the following CUDA kernel:

Cuda Code:
```cuda
{cuda_code}
```

Cpp Code:
"""
