import re
from loguru import logger


def remove_code_block_lines(text):
    return "\n".join([line for line in text.split("\n") if not line.strip().startswith("```")])


def replace_wrapper_func_first_arg(wrapper_code: str, to_replace: str) -> str:
    """
    wrapper(set_sorting_offset, nrows1, ncols1, offsets1); = >  wrapper(to_replace, nrows1, ncols1, offsets1);
    """
    pattern = r"wrapper\((\w+),"
    try:
        res = re.sub(pattern, f"wrapper({to_replace},", wrapper_code)
        return res
    except:
        logger.error(f"replace_wrapper_func_first_arg Error: {wrapper_code}")
        return wrapper_code


def replace_kernel_function_in_wrapper(wrapper_code: str, to_function_name: str) -> str:
    # 修改正则表达式以处理函数名前后的空格
    # \s* 匹配任意数量的空白字符
    pattern = r'(\w+)\s*(?=\s*<<<)'
    return re.sub(pattern, to_function_name, wrapper_code)


def get_function_name_from_cpp_or_cuda_code(code: str) -> str:
    if not code or code == " ":
        return " "
    # TODO: cuda case maybe need to check further
    # remove the \n
    replaced_code = code.replace("\n", " ")
    front_code = replaced_code.split("(")[0].strip(" ")
    try:
        res = front_code.split()[-1]
    except:
        logger.error(f"get_function_name_from_cpp_or_cuda_code Error: {code}")
        return " "
    if res:
        return res
    return " "


def remove_comments(code):
    # 删除单行注释
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    # 删除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code


def replace_assert_statements(code: str) -> str:
    """
    Match and replace assert statements that end with semicolon.
    Example: "assert x > 0;" -> ""
    """

    # Pattern matches 'assert' followed by any characters until semicolon
    pattern = r"assert[^;]*;"
    # Replace all matching patterns with empty string
    result = re.sub(pattern, "", code)
    return result.strip()


def replace_cuda_free_statements(code: str) -> str:
    pattern = r"cudaFree\([^;]*\);"
    result = re.sub(pattern, "", code)
    return result


def wrapper_function_invoke_to_print_variables(code: str, function_name: str) -> str:
    """
    wrongfunc(args) => wrapper(func, args), func(args) is the last line of the code
    """
    # Pattern matches: function_name(any_params);
    pattern = r"(\w+)\s*\(([^;]*?)\);$"

    lines = code.strip().split('\n')
    last_line = lines[-1]
    match = re.search(pattern, last_line)
    if match:
        params_str = match.group(2)
        res = f"wrapper({function_name}, {params_str});"
        return "\n".join(lines[:-1] + [res])
    raise ValueError(f"Function call not found in the last line: {last_line}")


def replace_wrapper_invoke(code: str) -> str:
    """
    Replace all wrapper function calls with direct function calls:
    wrapper(func, arg1, arg2, ...) => func(arg1. arg2, ...)
    """
    def replacement(match):
        wrapper_call = match.group()
        params_str = re.search(r"wrapper\((\w+),\s*(.*)\)", wrapper_call)
        if params_str:
            function_name = params_str.group(1)
            params = params_str.group(2).strip()
            return f"{function_name}({params});"
        return wrapper_call

    pattern = r"wrapper\([^;]*?\);"
    return re.sub(pattern, replacement, code)


def replace_wrapper_invoke_back(code: str, function_name) -> str:
    """
    Replace all wrapper function calls with direct function calls:
    wrapper(func, args) => func(args)
    """
    def replacement(match):
        wrapper_call = match.group()
        params_str = re.search(
            rf"wrapper\({function_name},\s*(.*)\)", wrapper_call)
        if params_str:
            params = params_str.group(1).strip()
            return f"{function_name}({params});"
        return wrapper_call

    pattern = rf"wrapper\({function_name},[^;]*?\);"
    return re.sub(pattern, replacement, code)
