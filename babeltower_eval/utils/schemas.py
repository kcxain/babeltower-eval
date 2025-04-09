from pydantic import BaseModel
from typing import List, Optional

from babeltower_eval.templates import CPP_UNITTEST_TEMPLATES, CUDA_UNITTEST_TEMPLATES, CPP_UNITTEST_TEMPLATES_FOR_COV


class TransDirection:
    source: str
    target: str

    def __init__(self, source: Optional[str] = None, target: Optional[str] = None, **kwargs):
        trans_pair = {
            "CUDA": "CPP",
            "CPP": "CUDA",
        }
        if source:
            self.source = source.upper()
            self.target = trans_pair[source.upper()]
        elif target:
            self.target = target.upper()
            self.source = trans_pair[target.upper()]
        else:
            raise ValueError("Either source or target must be provided.")


class UnitTestEvalCase(BaseModel):
    id: int
    cpp_code: str
    cuda_code: str
    consistent_cpp_inputs: List[str]
    consistent_cuda_inputs: List[str]
    cuda_wrapper: str
    source: str = "None"

    def format_cuda_code(self) -> List[str]:
        cuda_unittest_codes = []
        for test_case in self.consistent_cuda_inputs:
            code = CUDA_UNITTEST_TEMPLATES.replace(
                "// KERNEL_FUNC", self.cuda_code)
            code = code.replace("// WRAPPER_FUNC", self.cuda_wrapper)
            code = code.replace("// TEST_CASE", test_case)
            cuda_unittest_codes.append(code)
        return cuda_unittest_codes

    def format_cpp_code(self) -> List[str]:
        cpp_unittest_codes = []
        for test_case in self.consistent_cpp_inputs:
            code = CPP_UNITTEST_TEMPLATES.replace(
                "// TO_FILL_FUNC", self.cpp_code)
            code = code.replace("// TEST_CASE", test_case)
            cpp_unittest_codes.append(code)
        return cpp_unittest_codes
