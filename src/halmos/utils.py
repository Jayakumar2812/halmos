# SPDX-License-Identifier: AGPL-3.0

import math
import re
import uuid
from functools import partial
from timeit import default_timer as timer
from typing import Any

from z3 import (
    Z3_OP_BADD,
    Z3_OP_CONCAT,
    Z3_OP_ULEQ,
    BitVec,
    BitVecNumRef,
    BitVecRef,
    BitVecSort,
    BitVecVal,
    BoolRef,
    BoolVal,
    Concat,
    Extract,
    Function,
    If,
    Not,
    Or,
    SignExt,
    SolverFor,
    eq,
    is_app,
    is_app_of,
    is_bool,
    is_bv,
    is_bv_value,
    is_const,
    is_not,
    simplify,
    substitute,
)

from .bitvec import HalmosBitVec as BV
from .bitvec import HalmosBool as Bool
from .exceptions import HalmosException, NotConcreteError
from .logs import warn
from .mapper import Mapper

# order of the secp256k1 curve
secp256k1n = (
    115792089237316195423570985008687907852837564279074904382605163141518161494337
)

Byte = int | BitVecRef | BV  # uint8
Bytes4 = int | BitVecRef | BV  # uint32
Address = int | BitVecRef | BV  # uint160
Word = int | BitVecRef | BV  # uint256
Bytes = "bytes | BitVecRef | ByteVec"  # arbitrary-length sequence of bytes


# dynamic BitVecSort sizes
class BitVecSortCache:
    def __init__(self):
        self.cache = {}
        for size in (
            1,
            8,
            16,
            32,
            64,
            128,
            160,
            256,
            264,
            288,
            512,
            544,
            800,
            1024,
            1056,
        ):
            self.cache[size] = BitVecSort(size)

    def __getitem__(self, size: int) -> BitVecSort:
        hit = self.cache.get(size)
        return hit if hit is not None else BitVecSort(size)


BitVecSorts = BitVecSortCache()

# known, fixed BitVecSort sizes
BitVecSort1 = BitVecSorts[1]
BitVecSort8 = BitVecSorts[8]
BitVecSort160 = BitVecSorts[160]
BitVecSort256 = BitVecSorts[256]
BitVecSort264 = BitVecSorts[264]
BitVecSort512 = BitVecSorts[512]


# ecrecover(digest, v, r, s)
f_ecrecover = Function(
    "f_ecrecover",
    BitVecSort256,
    BitVecSort8,
    BitVecSort256,
    BitVecSort256,
    BitVecSort160,
)


def is_f_sha3_name(name: str) -> bool:
    return name.startswith("f_sha3_")


def f_sha3_name(bitsize: int) -> str:
    return f"f_sha3_{bitsize}"


def f_inv_sha3_name(bitsize: int) -> str:
    return f"f_inv_sha3_{bitsize}"


# TODO: explore the impact of using a smaller bitsize for the range sort
f_inv_sha3_size = Function("f_inv_sha3_size", BitVecSort160, BitVecSort256)


f_sha3_0_name = f_sha3_name(0)
f_sha3_256_name = f_sha3_name(256)
f_sha3_512_name = f_sha3_name(512)

# NOTE: another way to encode the empty keccak is to use 0-ary function like:
#         f_sha3_empty = Function(f_sha3_0_name, BitVecSort256)
#       then `f_sha3_empty()` is equivalent to `BitVec(f_sha3_0_name, BitVecSort256)`.
#       in both cases, decl() == f_sha3_0_name, and num_args() == 0.
f_sha3_empty = BitVec(f_sha3_0_name, BitVecSort256)


def uid() -> str:
    return uuid.uuid4().hex[:7]


def wrap(x: Any) -> Word:
    if is_bv(x):
        return x
    if isinstance(x, int):
        return con(x)
    if isinstance(x, bytes):
        return BitVecVal(int.from_bytes(x, "big"), 8 * len(x))
    raise ValueError(x)


def concat(args):
    if len(args) > 1:
        return Concat([wrap(x) for x in args])
    else:
        return args[0]


def smt_or(args):
    if len(args) > 1:
        return Or(args)
    else:
        return args[0]


def uint(x: Any, n: int) -> Word:
    """
    Truncates or zero-extends x to n bits
    """

    return BV(x, size=n)


def uint8(x: Any) -> Byte:
    return uint(x, 8)


def uint160(x: Word) -> Address:
    return uint(x, 160)


def uint256(x: Any) -> Word:
    return uint(x, 256)


def int256(x: BitVecRef) -> BitVecRef:
    if isinstance(x, int):
        return con(x, size_bits=256)

    if is_bool(x):
        return If(x, con(1, size_bits=256), con(0, size_bits=256))

    bitsize = x.size()
    if bitsize > 256:
        raise ValueError(x)
    if bitsize == 256:
        return x
    return simplify(SignExt(256 - bitsize, x))


def address(x: Any) -> Address:
    return uint(x, 160)


def con(n: int, size_bits=256) -> Word:
    return BitVecVal(n, BitVecSorts[size_bits])


def z3_bv(x: Any) -> BitVecRef:
    if isinstance(x, BV):
        return x.as_z3()

    if isinstance(x, Bool):
        return BV(x).as_z3()

    # must check before int because isinstance(True, int) is True
    if isinstance(x, bool):
        return BoolVal(x)

    if isinstance(x, int):
        return con(x, size_bits=256)

    if is_bv(x) or is_bool(x):
        return x

    raise ValueError(x)


#             x  == b   if sort(x) = bool
# int_to_bool(x) == b   if sort(x) = int
def test(x: Word, b: bool) -> BoolRef:
    if isinstance(x, int):
        return BoolVal(x != 0) if b else BoolVal(x == 0)

    elif isinstance(x, BV):
        return x.is_non_zero().as_z3() if b else x.is_zero().as_z3()

    elif is_bool(x):
        if b:
            return x
        else:
            return Not(x)

    elif is_bv(x):
        return x != con(0) if b else x == con(0)

    else:
        raise ValueError(x)


def is_concrete(x: Any) -> bool:
    if isinstance(x, BV | Bool):
        return x.is_concrete

    return isinstance(x, int | bytes) or is_bv_value(x)


def is_concat(x: BitVecRef) -> bool:
    return is_app_of(x, Z3_OP_CONCAT)


def create_solver(logic="QF_AUFBV", ctx=None, timeout=0, max_memory=0):
    # QF_AUFBV: quantifier-free bitvector + array theory: https://smtlib.cs.uiowa.edu/logics.shtml
    solver = SolverFor(logic, ctx=ctx)

    # set timeout
    solver.set(timeout=timeout)

    # set memory limit
    if max_memory > 0:
        solver.set(max_memory=max_memory)

    return solver


def extract_bytes32_array_argument(data: Bytes, arg_idx: int):
    """Extracts idx-th argument of bytes32[] from calldata"""
    offset = int_of(
        extract_bytes(data, 4 + arg_idx * 32, 32),
        "symbolic offset for bytes argument",
    )
    length = int_of(
        extract_bytes(data, 4 + offset, 32),
        "symbolic size for bytes argument",
    )
    if length == 0:
        return b""

    return extract_bytes(data, 4 + offset + 32, length * 32)


def extract_bytes_argument(data: Bytes, arg_idx: int) -> bytes:
    """Extracts idx-th argument of string from data"""
    offset = int_of(
        extract_word(data, 4 + arg_idx * 32), "symbolic offset for bytes argument"
    )
    length = int_of(extract_word(data, 4 + offset), "symbolic size for bytes argument")
    if length == 0:
        return b""

    bytes = extract_bytes(data, 4 + offset + 32, length)
    return bv_value_to_bytes(bytes) if is_bv_value(bytes) else bytes


def extract_string_argument(data: Bytes, arg_idx: int):
    """Extracts idx-th argument of string from data"""
    string_bytes = extract_bytes_argument(data, arg_idx)
    return string_bytes.decode("utf-8") if is_concrete(string_bytes) else string_bytes


def extract_bytes(data: Bytes, offset: int, size_bytes: int) -> Bytes:
    """Extract bytes from data. Zero-pad if out of bounds."""
    if hasattr(data, "__getitem__"):
        data_slice = data[offset : offset + size_bytes]
        return data_slice.unwrap() if hasattr(data_slice, "unwrap") else data_slice

    if data is None:
        return BitVecVal(0, size_bytes * 8)

    n = data.size()
    if n % 8 != 0:
        raise ValueError(n)

    # will extract hi - lo + 1 bits
    hi = n - 1 - offset * 8
    lo = n - offset * 8 - size_bytes * 8
    lo = 0 if lo < 0 else lo

    val = simplify(Extract(hi, lo, data))

    zero_padding = size_bytes * 8 - val.size()
    if zero_padding < 0:
        raise ValueError(val)
    if zero_padding > 0:
        val = simplify(Concat(val, con(0, zero_padding)))

    return val


def extract_word(data: Bytes, offset: int) -> Word:
    """Extracts a 256-bit word from data at offset"""
    return extract_bytes(data, offset, 32)


def extract_funsig(data: Bytes) -> Bytes4:
    """Extracts the function signature (first 4 bytes) from calldata"""
    if hasattr(data, "__getitem__"):
        return unbox_int(data[:4])
    return extract_bytes(data, 0, 4)


def bv_value_to_bytes(x: BitVecNumRef) -> bytes:
    return x.as_long().to_bytes(byte_length(x, strict=True), "big")


def try_bv_value_to_bytes(x: Any) -> Any:
    return bv_value_to_bytes(x) if is_bv_value(x) else x


def bytes_to_bv_value(x: bytes) -> BitVecNumRef:
    return con(int.from_bytes(x, "big"), size_bits=len(x) * 8)


def unbox_int(x: Any) -> Any:
    """
    Converts int-like objects to int, returns x otherwise
    """
    if isinstance(x, int):
        return x

    if hasattr(x, "unwrap"):
        return unbox_int(x.unwrap())

    if isinstance(x, bytes):
        return int.from_bytes(x, "big")

    if is_bv_value(x):
        return x.as_long()

    return x


def int_of(x: Any, err: str = None, subst: dict = None) -> int:
    """
    Converts int-like objects to int or raises NotConcreteError
    """

    if hasattr(x, "unwrap"):
        x = x.unwrap()

    # attempt to replace symbolic (sub-)terms with their concrete values
    if subst and is_bv(x) and not is_bv_value(x):
        x = simplify(substitute(x, *subst.items()))

    res = unbox_int(x)

    if isinstance(res, int):
        return res

    err = err or "expected concrete value but got"
    raise NotConcreteError(f"{err}: {x}")


def byte_length(x: Any, strict=True) -> int:
    if hasattr(x, "__len__"):
        # bytes, lists, tuples, bytevecs, chunks...
        return len(x)

    if is_bv(x):
        if x.size() % 8 != 0 and strict:
            raise HalmosException(f"byte_length({x}) with bit size {x.size()}")
        return math.ceil(x.size() / 8)

    raise TypeError(f"byte_length({x}) of type {type(x)}")


def match_dynamic_array_overflow_condition(cond: BitVecRef) -> bool:
    """
    Check if `cond` matches the following pattern:
        Not(ULE(f_sha3_N(slot), offset + f_sha3_N(slot))), where offset < 2**64

    This condition is satisfied when a dynamic array at `slot` exceeds the storage limit.
    Since such an overflow is highly unlikely in practice, we assume that this condition is unsat.

    Note: we already assume that any sha3 hash output is smaller than 2**256 - 2**64 (see SEVM.sha3_data()).
    However, the smt solver may not be able to solve this condition within the branching timeout.
    In such cases, this explicit pattern serves as a fallback to avoid exploring practically infeasible paths.

    We don't need to handle the negation of this condition, because unknown conditions are conservatively assumed to be sat.
    """

    # Not(ule)
    if not is_not(cond):
        return False
    ule = cond.arg(0)

    # Not(ULE(left, right)
    if not is_app_of(ule, Z3_OP_ULEQ):
        return False
    left, right = ule.arg(0), ule.arg(1)

    # Not(ULE(f_sha3_N(slot), offset + base))
    if not (is_f_sha3_name(left.decl().name()) and is_app_of(right, Z3_OP_BADD)):
        return False
    offset, base = right.arg(0), right.arg(1)

    # Not(ULE(f_sha3_N(slot), offset + f_sha3_N(slot))) and offset < 2**64
    return eq(left, base) and is_bv_value(offset) and offset.as_long() < 2**64


def stripped(hexstring: str) -> str:
    """Remove 0x prefix from hexstring"""
    return hexstring[2:] if hexstring.startswith("0x") else hexstring


def decode_hex(hexstring: str) -> bytes | None:
    try:
        # not checking if length is even because fromhex accepts spaces
        return bytes.fromhex(stripped(hexstring))
    except ValueError:
        return None


def hexify(x, contract_name: str = None):
    if isinstance(x, str):
        return re.sub(r"\b(\d+)\b", lambda match: hex(int(match.group(1))), x)
    elif isinstance(x, int):
        return f"0x{x:02x}"
    elif isinstance(x, bytes):
        return Mapper().lookup_selector("0x" + x.hex(), contract_name)
    elif hasattr(x, "unwrap"):
        return hexify(x.unwrap(), contract_name)
    elif is_bv_value(x):
        # maintain the byte size of x
        num_bytes = byte_length(x, strict=False)
        return Mapper().lookup_selector(
            f"0x{x.as_long():0{num_bytes * 2}x}", contract_name
        )
    elif is_app(x):
        params_and_children = (
            f"({', '.join(map(partial(hexify, contract_name=contract_name), x.params() + x.children()))})"
            if not is_const(x)
            else ""
        )
        return f"{str(x.decl())}{params_and_children}"
    else:
        return hexify(str(x), contract_name)


def render_uint(x: BitVecRef) -> str:
    if is_bv_value(x):
        val = int_of(x)
        return f"0x{val:0{byte_length(x, strict=False) * 2}x} ({val})"

    return hexify(x)


def render_int(x: BitVecRef) -> str:
    if is_bv_value(x):
        val = x.as_signed_long()
        return f"0x{x.as_long():0{byte_length(x, strict=False) * 2}x} ({val})"

    return hexify(x)


def render_bool(b: BitVecRef) -> str:
    return str(b.as_long() != 0).lower() if is_bv_value(b) else hexify(b)


def render_string(s: BitVecRef) -> str:
    str_val = bytes.fromhex(stripped(hexify(s))).decode("utf-8")
    return f'"{str_val}"'


def render_bytes(b: Bytes) -> str:
    if is_bv(b):
        return hexify(b) + f" ({byte_length(b, strict=False)} bytes)"
    else:
        return f'hex"{stripped(b.hex())}"'


def render_address(a: BitVecRef) -> str:
    if is_bv_value(a):
        return f"0x{a.as_long():040x}"

    return hexify(a)


def stringify(symbol_name: str, val: Any):
    """
    Formats a value based on the inferred type of the variable.

    Expects symbol_name to be of the form 'p_<sourceVar>_<sourceType>', e.g. 'p_x_uint256'
    """
    if not is_bv_value(val):
        warn(f"{val} is not a bitvector value")
        return hexify(val)

    tokens = symbol_name.split("_")
    if len(tokens) < 3:
        warn(f"Failed to infer type for symbol '{symbol_name}'")
        return hexify(val)

    if len(tokens) >= 4 and tokens[-1].isdigit():
        # we may have something like p_val_bytes_01
        # the last token being a symbol number, discard it
        tokens.pop()

    type_name = tokens[-1]

    try:
        if type_name.startswith("uint"):
            return render_uint(val)
        elif type_name.startswith("int"):
            return render_int(val)
        elif type_name == "bool":
            return render_bool(val)
        elif type_name == "string":
            return render_string(val)
        elif type_name == "bytes":
            return render_bytes(val)
        elif type_name == "address":
            return render_address(val)
        else:  # bytes32, bytes4, structs, etc.
            return hexify(val)
    except Exception as e:
        # log error and move on
        warn(f"Failed to stringify {val} of type {type_name}: {repr(e)}")
        return hexify(val)


def assert_bv(x) -> None:
    if not is_bv(x):
        raise ValueError(x)


def assert_address(x: Word) -> None:
    if isinstance(x, BV):
        if x.size != 160:
            raise ValueError(x)
        return

    if is_concrete(x):
        if not 0 <= int_of(x) < 2**160:
            raise ValueError(x)
        return

    if x.size() != 160:
        raise ValueError(x)


def assert_uint256(x: Word) -> None:
    if isinstance(x, BV):
        if x.size != 256:
            raise ValueError(x)
        return

    if is_concrete(x):
        if not 0 <= int_of(x) < 2**256:
            raise ValueError(x)
        return

    if x.size() != 256:
        raise ValueError(x)


def con_addr(n: int) -> BitVecRef:
    if n >= 2**160:
        raise ValueError(n)
    return BitVecVal(n, 160)


def green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m"


def magenta(text: str) -> str:
    return f"\033[95m{text}\033[0m"


color_good = green
color_debug = magenta
color_info = cyan
color_warn = yellow
color_error = red


def indent_text(text: str, n: int = 4) -> str:
    return "\n".join(" " * n + line for line in text.splitlines())


class EVM:
    STOP = 0x00
    ADD = 0x01
    MUL = 0x02
    SUB = 0x03
    DIV = 0x04
    SDIV = 0x05
    MOD = 0x06
    SMOD = 0x07
    ADDMOD = 0x08
    MULMOD = 0x09
    EXP = 0x0A
    SIGNEXTEND = 0x0B
    LT = 0x10
    GT = 0x11
    SLT = 0x12
    SGT = 0x13
    EQ = 0x14
    ISZERO = 0x15
    AND = 0x16
    OR = 0x17
    XOR = 0x18
    NOT = 0x19
    BYTE = 0x1A
    SHL = 0x1B
    SHR = 0x1C
    SAR = 0x1D
    SHA3 = 0x20
    ADDRESS = 0x30
    BALANCE = 0x31
    ORIGIN = 0x32
    CALLER = 0x33
    CALLVALUE = 0x34
    CALLDATALOAD = 0x35
    CALLDATASIZE = 0x36
    CALLDATACOPY = 0x37
    CODESIZE = 0x38
    CODECOPY = 0x39
    GASPRICE = 0x3A
    EXTCODESIZE = 0x3B
    EXTCODECOPY = 0x3C
    RETURNDATASIZE = 0x3D
    RETURNDATACOPY = 0x3E
    EXTCODEHASH = 0x3F
    BLOCKHASH = 0x40
    COINBASE = 0x41
    TIMESTAMP = 0x42
    NUMBER = 0x43
    DIFFICULTY = 0x44
    GASLIMIT = 0x45
    CHAINID = 0x46
    SELFBALANCE = 0x47
    BASEFEE = 0x48
    POP = 0x50
    MLOAD = 0x51
    MSTORE = 0x52
    MSTORE8 = 0x53
    SLOAD = 0x54
    SSTORE = 0x55
    JUMP = 0x56
    JUMPI = 0x57
    PC = 0x58
    MSIZE = 0x59
    GAS = 0x5A
    JUMPDEST = 0x5B
    TLOAD = 0x5C
    TSTORE = 0x5D
    MCOPY = 0x5E
    PUSH0 = 0x5F
    PUSH1 = 0x60
    PUSH2 = 0x61
    PUSH3 = 0x62
    PUSH4 = 0x63
    PUSH5 = 0x64
    PUSH6 = 0x65
    PUSH7 = 0x66
    PUSH8 = 0x67
    PUSH9 = 0x68
    PUSH10 = 0x69
    PUSH11 = 0x6A
    PUSH12 = 0x6B
    PUSH13 = 0x6C
    PUSH14 = 0x6D
    PUSH15 = 0x6E
    PUSH16 = 0x6F
    PUSH17 = 0x70
    PUSH18 = 0x71
    PUSH19 = 0x72
    PUSH20 = 0x73
    PUSH21 = 0x74
    PUSH22 = 0x75
    PUSH23 = 0x76
    PUSH24 = 0x77
    PUSH25 = 0x78
    PUSH26 = 0x79
    PUSH27 = 0x7A
    PUSH28 = 0x7B
    PUSH29 = 0x7C
    PUSH30 = 0x7D
    PUSH31 = 0x7E
    PUSH32 = 0x7F
    DUP1 = 0x80
    DUP2 = 0x81
    DUP3 = 0x82
    DUP4 = 0x83
    DUP5 = 0x84
    DUP6 = 0x85
    DUP7 = 0x86
    DUP8 = 0x87
    DUP9 = 0x88
    DUP10 = 0x89
    DUP11 = 0x8A
    DUP12 = 0x8B
    DUP13 = 0x8C
    DUP14 = 0x8D
    DUP15 = 0x8E
    DUP16 = 0x8F
    SWAP1 = 0x90
    SWAP2 = 0x91
    SWAP3 = 0x92
    SWAP4 = 0x93
    SWAP5 = 0x94
    SWAP6 = 0x95
    SWAP7 = 0x96
    SWAP8 = 0x97
    SWAP9 = 0x98
    SWAP10 = 0x99
    SWAP11 = 0x9A
    SWAP12 = 0x9B
    SWAP13 = 0x9C
    SWAP14 = 0x9D
    SWAP15 = 0x9E
    SWAP16 = 0x9F
    LOG0 = 0xA0
    LOG1 = 0xA1
    LOG2 = 0xA2
    LOG3 = 0xA3
    LOG4 = 0xA4
    CREATE = 0xF0
    CALL = 0xF1
    CALLCODE = 0xF2
    RETURN = 0xF3
    DELEGATECALL = 0xF4
    CREATE2 = 0xF5
    STATICCALL = 0xFA
    REVERT = 0xFD
    INVALID = 0xFE
    SELFDESTRUCT = 0xFF


str_opcode: dict[int, str] = {
    EVM.STOP: "STOP",
    EVM.ADD: "ADD",
    EVM.MUL: "MUL",
    EVM.SUB: "SUB",
    EVM.DIV: "DIV",
    EVM.SDIV: "SDIV",
    EVM.MOD: "MOD",
    EVM.SMOD: "SMOD",
    EVM.ADDMOD: "ADDMOD",
    EVM.MULMOD: "MULMOD",
    EVM.EXP: "EXP",
    EVM.SIGNEXTEND: "SIGNEXTEND",
    EVM.LT: "LT",
    EVM.GT: "GT",
    EVM.SLT: "SLT",
    EVM.SGT: "SGT",
    EVM.EQ: "EQ",
    EVM.ISZERO: "ISZERO",
    EVM.AND: "AND",
    EVM.OR: "OR",
    EVM.XOR: "XOR",
    EVM.NOT: "NOT",
    EVM.BYTE: "BYTE",
    EVM.SHL: "SHL",
    EVM.SHR: "SHR",
    EVM.SAR: "SAR",
    EVM.SHA3: "SHA3",
    EVM.ADDRESS: "ADDRESS",
    EVM.BALANCE: "BALANCE",
    EVM.ORIGIN: "ORIGIN",
    EVM.CALLER: "CALLER",
    EVM.CALLVALUE: "CALLVALUE",
    EVM.CALLDATALOAD: "CALLDATALOAD",
    EVM.CALLDATASIZE: "CALLDATASIZE",
    EVM.CALLDATACOPY: "CALLDATACOPY",
    EVM.CODESIZE: "CODESIZE",
    EVM.CODECOPY: "CODECOPY",
    EVM.GASPRICE: "GASPRICE",
    EVM.EXTCODESIZE: "EXTCODESIZE",
    EVM.EXTCODECOPY: "EXTCODECOPY",
    EVM.RETURNDATASIZE: "RETURNDATASIZE",
    EVM.RETURNDATACOPY: "RETURNDATACOPY",
    EVM.EXTCODEHASH: "EXTCODEHASH",
    EVM.BLOCKHASH: "BLOCKHASH",
    EVM.COINBASE: "COINBASE",
    EVM.TIMESTAMP: "TIMESTAMP",
    EVM.NUMBER: "NUMBER",
    EVM.DIFFICULTY: "DIFFICULTY",
    EVM.GASLIMIT: "GASLIMIT",
    EVM.CHAINID: "CHAINID",
    EVM.SELFBALANCE: "SELFBALANCE",
    EVM.BASEFEE: "BASEFEE",
    EVM.POP: "POP",
    EVM.MCOPY: "MCOPY",
    EVM.MLOAD: "MLOAD",
    EVM.MSTORE: "MSTORE",
    EVM.MSTORE8: "MSTORE8",
    EVM.SLOAD: "SLOAD",
    EVM.SSTORE: "SSTORE",
    EVM.JUMP: "JUMP",
    EVM.JUMPI: "JUMPI",
    EVM.PC: "PC",
    EVM.MSIZE: "MSIZE",
    EVM.GAS: "GAS",
    EVM.JUMPDEST: "JUMPDEST",
    EVM.TLOAD: "TLOAD",
    EVM.TSTORE: "TSTORE",
    EVM.MCOPY: "MCOPY",
    EVM.PUSH0: "PUSH0",
    EVM.PUSH1: "PUSH1",
    EVM.PUSH2: "PUSH2",
    EVM.PUSH3: "PUSH3",
    EVM.PUSH4: "PUSH4",
    EVM.PUSH5: "PUSH5",
    EVM.PUSH6: "PUSH6",
    EVM.PUSH7: "PUSH7",
    EVM.PUSH8: "PUSH8",
    EVM.PUSH9: "PUSH9",
    EVM.PUSH10: "PUSH10",
    EVM.PUSH11: "PUSH11",
    EVM.PUSH12: "PUSH12",
    EVM.PUSH13: "PUSH13",
    EVM.PUSH14: "PUSH14",
    EVM.PUSH15: "PUSH15",
    EVM.PUSH16: "PUSH16",
    EVM.PUSH17: "PUSH17",
    EVM.PUSH18: "PUSH18",
    EVM.PUSH19: "PUSH19",
    EVM.PUSH20: "PUSH20",
    EVM.PUSH21: "PUSH21",
    EVM.PUSH22: "PUSH22",
    EVM.PUSH23: "PUSH23",
    EVM.PUSH24: "PUSH24",
    EVM.PUSH25: "PUSH25",
    EVM.PUSH26: "PUSH26",
    EVM.PUSH27: "PUSH27",
    EVM.PUSH28: "PUSH28",
    EVM.PUSH29: "PUSH29",
    EVM.PUSH30: "PUSH30",
    EVM.PUSH31: "PUSH31",
    EVM.PUSH32: "PUSH32",
    EVM.DUP1: "DUP1",
    EVM.DUP2: "DUP2",
    EVM.DUP3: "DUP3",
    EVM.DUP4: "DUP4",
    EVM.DUP5: "DUP5",
    EVM.DUP6: "DUP6",
    EVM.DUP7: "DUP7",
    EVM.DUP8: "DUP8",
    EVM.DUP9: "DUP9",
    EVM.DUP10: "DUP10",
    EVM.DUP11: "DUP11",
    EVM.DUP12: "DUP12",
    EVM.DUP13: "DUP13",
    EVM.DUP14: "DUP14",
    EVM.DUP15: "DUP15",
    EVM.DUP16: "DUP16",
    EVM.SWAP1: "SWAP1",
    EVM.SWAP2: "SWAP2",
    EVM.SWAP3: "SWAP3",
    EVM.SWAP4: "SWAP4",
    EVM.SWAP5: "SWAP5",
    EVM.SWAP6: "SWAP6",
    EVM.SWAP7: "SWAP7",
    EVM.SWAP8: "SWAP8",
    EVM.SWAP9: "SWAP9",
    EVM.SWAP10: "SWAP10",
    EVM.SWAP11: "SWAP11",
    EVM.SWAP12: "SWAP12",
    EVM.SWAP13: "SWAP13",
    EVM.SWAP14: "SWAP14",
    EVM.SWAP15: "SWAP15",
    EVM.SWAP16: "SWAP16",
    EVM.LOG0: "LOG0",
    EVM.LOG1: "LOG1",
    EVM.LOG2: "LOG2",
    EVM.LOG3: "LOG3",
    EVM.LOG4: "LOG4",
    EVM.CREATE: "CREATE",
    EVM.CALL: "CALL",
    EVM.CALLCODE: "CALLCODE",
    EVM.RETURN: "RETURN",
    EVM.DELEGATECALL: "DELEGATECALL",
    EVM.CREATE2: "CREATE2",
    EVM.STATICCALL: "STATICCALL",
    EVM.REVERT: "REVERT",
    EVM.INVALID: "INVALID",
    EVM.SELFDESTRUCT: "SELFDESTRUCT",
}


def restore_precomputed_hashes(x: int) -> tuple[int, int]:
    (preimage, offset) = sha3_inv_offset.get(x >> 16, (None, None))
    if preimage is None:
        return (None, None)
    delta = (x & 0xFFFF) - offset
    if delta < 0:
        return (None, None)
    return (preimage, delta)  # x == hash(preimage) + delta


def mk_sha3_inv_offset(m: dict[int, int]) -> dict[int, tuple[int, int]]:
    m2 = {}
    for k, v in m.items():
        m2[k >> 16] = (v, k & 0xFFFF)
    return m2


sha3_inv: dict[int, int] = {  # sha3(x) -> x
    0x290DECD9548B62A8D60345A988386FC84BA6BC95484008F6362F93160EF3E563: 0,
    0xB10E2D527612073B26EECDFD717E6A320CF44B4AFAC2B0732D9FCBE2B7FA0CF6: 1,
    0x405787FA12A823E0F2B7631CC41B3BA8828B3321CA811111FA75CD3AA3BB5ACE: 2,
    0xC2575A0E9E593C00F959F8C92F12DB2869C3395A3B0502D05E2516446F71F85B: 3,
    0x8A35ACFBC15FF81A39AE7D344FD709F28E8600B4AA8C65C6B64BFE7FE36BD19B: 4,
    0x036B6384B5ECA791C62761152D0C79BB0604C104A5FB6F4EB0703F3154BB3DB0: 5,
    0xF652222313E28459528D920B65115C16C04F3EFC82AAEDC97BE59F3F377C0D3F: 6,
    0xA66CC928B5EDB82AF9BD49922954155AB7B0942694BEA4CE44661D9A8736C688: 7,
    0xF3F7A9FE364FAAB93B216DA50A3214154F22A0A2B415B23A84C8169E8B636EE3: 8,
    0x6E1540171B6C0C960B71A7020D9F60077F6AF931A8BBF590DA0223DACF75C7AF: 9,
    0xC65A7BB8D6351C1CF70C95A316CC6A92839C986682D98BC35F958F4883F9D2A8: 10,
    0x0175B7A638427703F0DBE7BB9BBF987A2551717B34E79F33B5B1008D1FA01DB9: 11,
    0xDF6966C971051C3D54EC59162606531493A51404A002842F56009D7E5CF4A8C7: 12,
    0xD7B6990105719101DABEB77144F2A3385C8033ACD3AF97E9423A695E81AD1EB5: 13,
    0xBB7B4A454DC3493923482F07822329ED19E8244EFF582CC204F8554C3620C3FD: 14,
    0x8D1108E10BCB7C27DDDFC02ED9D693A074039D026CF4EA4240B40F7D581AC802: 15,
    0x1B6847DC741A1B0CD08D278845F9D819D87B734759AFB55FE2DE5CB82A9AE672: 16,
    0x31ECC21A745E3968A04E9570E4425BC18FA8019C68028196B546D1669C200C68: 17,
    0xBB8A6A4669BA250D26CD7A459ECA9D215F8307E33AEBE50379BC5A3617EC3444: 18,
    0x66DE8FFDA797E3DE9C05E8FC57B3BF0EC28A930D40B0D285D93C06501CF6A090: 19,
    0xCE6D7B5282BD9A3661AE061FEED1DBDA4E52AB073B1F9285BE6E155D9C38D4EC: 20,
    0x55F448FDEA98C4D29EB340757EF0A66CD03DBB9538908A6A81D96026B71EC475: 21,
    0xD833147D7DC355BA459FC788F669E58CFAF9DC25DDCD0702E87D69C7B5124289: 22,
    0xC624B66CC0138B8FABC209247F72D758E1CF3343756D543BADBF24212BED8C15: 23,
    0xB13D2D76D1F4B7BE834882E410B3E3A8AFAF69F83600AE24DB354391D2378D2E: 24,
    0x944998273E477B495144FB8794C914197F3CCB46BE2900F4698FD0EF743C9695: 25,
    0x057C384A7D1C54F3A1B2E5E67B2617B8224FDFD1EA7234EEA573A6FF665FF63E: 26,
    0x3AD8AA4F87544323A9D1E5DD902F40C356527A7955687113DB5F9A85AD579DC1: 27,
    0x0E4562A10381DEC21B205ED72637E6B1B523BDD0E4D4D50AF5CD23DD4500A211: 28,
    0x6D4407E7BE21F808E6509AA9FA9143369579DD7D760FE20A2C09680FC146134F: 29,
    0x50BB669A95C7B50B7E8A6F09454034B2B14CF2B85C730DCA9A539CA82CB6E350: 30,
    0xA03837A25210EE280C2113FF4B77CA23440B19D4866CCA721C801278FD08D807: 31,
    0xC97BFAF2F8EE708C303A06D134F5ECD8389AE0432AF62DC132A24118292866BB: 32,
    0x3A6357012C1A3AE0A17D304C9920310382D968EBCC4B1771F41C6B304205B570: 33,
    0x61035B26E3E9EEE00E0D72FD1EE8DDCA6894550DCA6916EA2AC6BAA90D11E510: 34,
    0xD57B2B5166478FD4318D2ACC6CC2C704584312BDD8781B32D5D06ABDA57F4230: 35,
    0x7CD332D19B93BCABE3CCE7CA0C18A052F57E5FD03B4758A09F30F5DDC4B22EC4: 36,
    0x401968FF42A154441DA5F6C4C935AC46B8671F0E062BAAA62A7545BA53BB6E4C: 37,
    0x744A2CF8FD7008E3D53B67916E73460DF9FA5214E3EF23DD4259CA09493A3594: 38,
    0x98A476F1687BC3D60A2DA2ADBCBA2C46958E61FA2FB4042CD7BC5816A710195B: 39,
    0xE16DA923A2D88192E5070F37B4571D58682C0D66212EC634D495F33DE3F77AB5: 40,
    0xCB7C14CE178F56E2E8D86AB33EBC0AE081BA8556A00CD122038841867181CAAC: 41,
    0xBECED09521047D05B8960B7E7BCC1D1292CF3E4B2A6B63F48335CBDE5F7545D2: 42,
    0x11C44E4875B74D31FF9FD779BF2566AF7BD15B87FC985D01F5094B89E3669E4F: 43,
    0x7416C943B4A09859521022FD2E90EAC0DD9026DAD28FA317782A135F28A86091: 44,
    0x4A2CC91EE622DA3BC833A54C37FFCB6F3EC23B7793EFC5EAF5E71B7B406C5C06: 45,
    0x37FA166CBDBFBB1561CCD9EA985EC0218B5E68502E230525F544285B2BDF3D7E: 46,
    0xA813484AEF6FB598F9F753DAF162068FF39CCEA4075CB95E1A30F86995B5B7EE: 47,
    0x6FF97A59C90D62CC7236BA3A37CD85351BF564556780CF8C1157A220F31F0CBB: 48,
    0xC54045FA7C6EC765E825DF7F9E9BF9DEC12C5CEF146F93A5EEE56772EE647FBC: 49,
    0x11DF491316F14931039EDFD4F8964C9A443B862F02D4C7611D18C2BC4E6FF697: 50,
    0x82A75BDEEAE8604D839476AE9EFD8B0E15AA447E21BFD7F41283BB54E22C9A82: 51,
    0x46BDDB1178E94D7F2892FF5F366840EB658911794F2C3A44C450AA2C505186C1: 52,
    0xCFA4BEC1D3298408BB5AFCFCD9C430549C5B31F8AA5C5848151C0A55F473C34D: 53,
    0x4A11F94E20A93C79F6EC743A1954EC4FC2C08429AE2122118BF234B2185C81B8: 54,
    0x42A7B7DD785CD69714A189DFFB3FD7D7174EDC9ECE837694CE50F7078F7C31AE: 55,
    0x38395C5DCEADE9603479B177B68959049485DF8AA97B39F3533039AF5F456199: 56,
    0xDC16FEF70F8D5DDBC01EE3D903D1E69C18A3C7BE080EB86A81E0578814EE58D3: 57,
    0xA2999D817B6757290B50E8ECF3FA939673403DD35C97DE392FDB343B4015CE9E: 58,
    0xBBE3212124853F8B0084A66A2D057C2966E251E132AF3691DB153AB65F0D1A4D: 59,
    0xC6BB06CB7F92603DE181BF256CD16846B93B752A170FF24824098B31AA008A7E: 60,
    0xECE66CFDBD22E3F37D348A3D8E19074452862CD65FD4B9A11F0336D1AC6D1DC3: 61,
    0x8D800D6614D35EED73733EE453164A3B48076EB3138F466ADEEB9DEC7BB31F70: 62,
    0xC03004E3CE0784BF68186394306849F9B7B1200073105CD9AEB554A1802B58FD: 63,
    0x352FEEE0EEA125F11F791C1B77524172E9BC20F1B719B6CEF0FC24F64DB8E15E: 64,
    0x7C9785E8241615BC80415D89775984A1337D15DC1BF4CE50F41988B2A2B336A7: 65,
    0x38DFE4635B27BABECA8BE38D3B448CB5161A639B899A14825BA9C8D7892EB8C3: 66,
    0x9690AD99D6CE244EFA8A0F6C2D04036D3B33A9474DB32A71B71135C695102793: 67,
    0x9B22D3D61959B4D3528B1D8BA932C96FBE302B36A1AAD1D95CAB54F9E0A135EA: 68,
    0xA80A8FCC11760162F08BB091D2C9389D07F2B73D0E996161DFAC6F1043B5FC0B: 69,
    0x128667F541FED74A8429F9D592C26C2C6A4BEB9AE5EAD9912C98B2595C842310: 70,
    0xC43C1E24E1884C4E28A16BBD9506F60B5CA9F18FC90635E729D3CFE13ABCF001: 71,
    0x15040156076F78057C0A886F6DBAC29221FA3C2646ADBC8EFFEDAB98152FF32B: 72,
    0x37E472F504E93744DF80D87316862F9A8FD41A7BC266C723BF77DF7866D75F55: 73,
    0xFCC5BA1A98FC477B8948A04D08C6F4A76181FE75021370AB5E6ABD22B1792A2A: 74,
    0x17B0AF156A929EDF60C351F3DF2D53ED643FDD750AEF9EDA90DC7C8759A104A8: 75,
    0x42859D4F253F4D4A28EE9A59F9C9683A9404DA2C5D329C733AB84F150DB798A8: 76,
    0x1B524E1C8B5382BB913D0A2AAE8AD83BB92A45FCB47761FA4A12F5B6316C2B20: 77,
    0x9B65E484CE3D961A557081A44C6C68A0A27ECA0B88FCE820BDD99C3DC223DCC7: 78,
    0xA2E8F972DC9F7D0B76177BB8BE102E6BEC069EE42C61080745E8825470E80C6C: 79,
    0x5529612556959EF813DBE8D0ED29336AB75E80A9B7855030760B2917B01E568A: 80,
    0x994A4B4EDDB300691EE19901712848B1114BAD8A1A4AE195E5ABE0EC38021B94: 81,
    0xA9144A5E7EFD259B8B0D55467F4696ED47EC83317D61501B76366DBCCA65CE73: 82,
    0x4C83EFB3982AFBD500AB7C66D02B996DF5FDC3D20660E61600390AAD6D5F7F1E: 83,
    0xF0D642DBC7517672E217238A2F008F4F8CDAD0586D8CE5113E9E09DCC6860619: 84,
    0x71BEDA120AAFDD3BB922B360A066D10B7CE81D7AC2AD9874DAAC46E2282F6B45: 85,
    0xEA7419F5AE821E7204864E6A0871433BA612011908963BB42A64F42D65AD2F72: 86,
    0xE8E5595D268AAA85B36C3557E9D96C14A4FFFAEE9F45BCAE0C407968A7109630: 87,
    0x657000D47E971DCFB21375BCFA3496F47A2A2F0F12C8AEB78A008ACE6AE55CA5: 88,
    0xD73956B9E00D8F8BC5E44F7184DF1387CDD652E7726B8CCDA3DB4859E02F31BF: 89,
    0xE8C3ABD4193A84EC8A3FFF3EEB3ECBCBD0979E0C977AC1DEE06C6E01A60ACA1B: 90,
    0xFCEBC02DD307DC58CD01B156D63C6948B8F3422055FAC1D836349B01722E9C52: 91,
    0xEC0B854938343F85EB39A6648B9E449C2E4AEE4DC9B4E96AB592F9F497D05138: 92,
    0x2619EC68B255542E3DA68C054BFE0D7D0F27B7FDBEFC8BBCCDD23188FC71FE7F: 93,
    0x34D3C319F536DEB74ED8F1F3205D9AEFEF7487C819E77D3351630820DBFF1118: 94,
    0xCC7EE599E5D59FEE88C83157BD897847C5911DC7D317B3175E0B085198349973: 95,
    0x41C7AE758795765C6664A5D39BF63841C71FF191E9189522BAD8EBFF5D4ECA98: 96,
    0xF0ECB75DD1820844C57B6762233D4E26853B3A7B8157BBD9F41F280A0F1CEE9B: 97,
    0xB912C5EB6319A4A6A83580B9611610BEDB31614179330261BFD87A41347CAE1C: 98,
    0xD86D8A3F7C82C89ED8E04140017AA108A0A1469249F92C8F022B9DBAFA87B883: 99,
    0x26700E13983FEFBD9CF16DA2ED70FA5C6798AC55062A4803121A869731E308D2: 100,
    0x8FF97419363FFD7000167F130EF7168FBEA05FAF9251824CA5043F113CC6A7C7: 101,
    0x46501879B8CA8525E8C2FD519E2FBFCFA2EBEA26501294AA02CBFCFB12E94354: 102,
    0x9787EEB91FE3101235E4A76063C7023ECB40F923F97916639C598592FA30D6AE: 103,
    0xA2153420D844928B4421650203C77BABC8B33D7F2E7B450E2966DB0C22097753: 104,
    0x7FB4302E8E91F9110A6554C2C0A24601252C2A42C2220CA988EFCFE399914308: 105,
    0x116FEA137DB6E131133E7F2BAB296045D8F41CC5607279DB17B218CAB0929A51: 106,
    0xBD43CB8ECE8CD1863BCD6082D65C5B0D25665B1CE17980F0DA43C0ED545F98B4: 107,
    0x2B4A51AB505FC96A0952EFDA2BA61BCD3078D4C02C39A186EC16F21883FBE016: 108,
    0x5006B838207C6A9AE9B84D68F467DD4BB5C305FBFB6B04EAB8FAAABEEC1E18D8: 109,
    0x9930D9FF0DEE0EF5CA2F7710EA66B8F84DD0F5F5351ECFFE72B952CD9DB7142A: 110,
    0x39F2BABE526038520877FC7C33D81ACCF578AF4A06C5FA6B0D038CAE36E12711: 111,
    0x8F6B23FFA15F0465E3176E15CA644CF24F86DC1312FE715484E3C4AEAD5EB78B: 112,
    0xA1FCD19BFE8C32A61095B6BFBB2664842857E148FCBB5188386C8CD40348D5B6: 113,
    0xDFFBD64CC7C1A7EB27984335D9416D51137A03D3FABEC7141025C62663253FE1: 114,
    0xF79BDE9DDD17963EBCE6F7D021D60DE7C2BD0DB944D23C900C0C0E775F530052: 115,
    0x19A0B39AA25AC793B5F6E9A0534364CC0B3FD1EA9B651E79C7F50A59D48EF813: 116,
    0x9A8D93986A7B9E6294572EA6736696119C195C1A9F5EAE642D3C5FCD44E49DEA: 117,
    0xB5732705F5241370A28908C2FE1303CB223F03B90D857FD0573F003F79FEFED4: 118,
    0x7901CB5ADDCAE2D210A531C604A76A660D77039093BAC314DE0816A16392AFF1: 119,
    0x8DC6FB69531D98D70DC0420E638D2DFD04E09E1EC783EDE9AAC77DA9C5A0DAC4: 120,
    0x957BBDC7FAD0DEC56E7C96AF4A3AB63AA9DAF934A52FFCE891945B7FB622D791: 121,
    0xF0440771A29E57E18C66727944770B82CC77924AEF333C927CE6BDD2CDB3AE03: 122,
    0x5569044719A1EC3B04D0AFA9E7A5310C7C0473331D13DC9FAFE143B2C4E8148A: 123,
    0x9222CBF5D0DDC505A6F2F04716E22C226CEE16A955FEF88C618922096DAE2FD0: 124,
    0xA913C8AC5320DAE1C4A00FF23343947ED0FDF88D251E9BD2A5519D3D6162D222: 125,
    0x0F2ADA1F2DBAE48AE468FE0CDB7BCDA7D0CFFEE8545442E682273BA01A6203A7: 126,
    0x66925E85F1A4743FD8D60BA595ED74887B7CAF321DD83B21E04D77C115383408: 127,
    0x59F3FB058C6BBA7A4E76396639FC4DD21BD59163DB798899CF56CEF48B3C9EC9: 128,
    0x76FCE494794D92AC286B20D6126FC49ECB9CCA2FA94B5C726F6EC1109B891414: 129,
    0xB2244E644CFE16F72B654FBC48FF0FECEC8FC59649CA8625094BEBD9BD2E4035: 130,
    0x1397B88F412A83A7F1C0D834C533E486FF1F24F42A31819E91B624931060A863: 131,
    0x50250E93F8C73D2C1BE015EC28E8CD2FEB871EFA71E955AD24477AAFB09484FA: 132,
    0xDBDAEC72D84124D8C7C57AE448F5A4E3EEDB34DBA437FDCBE6D26496B68AFE87: 133,
    0x46B7EA84944250856A716737059479854246A026D947C13D5A0929BC8C1BC81D: 134,
    0x171AB08901BE24769DBEBEDBDF7E0245486FBC64AB975CD431A39533032D5415: 135,
    0x7EF464CF5A521D70C933977510816A0355B91A50ECA2778837FB82DA8448ECF6: 136,
    0x5BFA74C743914028161AE645D300D90BBDC659F169CA1469EC86B4960F7266CB: 137,
    0x834355D35CBFBD33B2397E201AF04B52BDD40B9B51275F279EA47E93547B631E: 138,
    0x7B6BB1E9D1B017FF82945596CF3CFB1A6CEE971C1EBB16F2C6BD23C2D642728E: 139,
    0x5F2F2DCA1D951C7429B52007F396328C64C25E226C1867318158F7F2CBDD40A9: 140,
    0x37A1BE2A88DADCD0E6062F54DDCC01A03360BA61CA7784A744E757488BF8CEB2: 141,
    0x8EDD81FF20324EA0CFE70C700FF4E9DB7580D269B423D9F61470B370819CBD17: 142,
    0x337F7913DB22D91EF425F82102BC8075EF67E23A2BE359965EA316E78E1EFF3F: 143,
    0x60B1E32550F9D5F25F9DD040E7A106B15D8EB282DD6B3E1914C73D8066896412: 144,
    0xCDAE184EDD6BF71C1FB62D6E6682FDB2032455C0E50143742135FBBE809BD793: 145,
    0x6E452848784197F00927D379E3DB9E69A5131D2269F862BFCD05A0B38F6ABF7F: 146,
    0x28DA5CA8143BFA5E9F642E58E5E87BEF0A2EB0C00BCD4EFDD01050293F5FAC91: 147,
    0x7047A3CC0A76EDCEE45792CA71527C753F6167484F14B94C4A3BD2997516725C: 148,
    0x947035E97D0F7E1937F791BC189F60C984CEAAA7A8494FC67F9F8F4DE8CCF2C6: 149,
    0x6AA7EC8AC2A999A90CE6C78668DFFE4E487E2576A97CA366EC81ECB335AF90D0: 150,
    0x354A83ED9988F79F6038D4C7A7DADBAD8AF32F4AD6DF893E0E5807A1B1944FF9: 151,
    0x2237A976FA961F5921FD19F2B03C925C725D77B20CE8F790C19709C03DE4D814: 152,
    0x72A152DDFB8E864297C917AF52EA6C1C68AEAD0FEE1A62673FCC7E0C94979D00: 153,
    0x44DA158BA27F9252712A74FF6A55C5D531F69609F1F6E7F17C4443A8E2089BE4: 154,
    0xBBA9DB4CDBEA0A37C207BBB83E20F828CD4441C49891101DC94FD20DC8EFC349: 155,
    0xAF85B9071DFAFEAC1409D3F1D19BAFC9BC7C37974CDE8DF0EE6168F0086E539C: 156,
    0xD26E832454299E9FABB89E0E5FFFDC046D4E14431BC1BF607FFB2E8A1DDECF7B: 157,
    0xCFE2A20FF701A1F3E14F63BD70D6C6BC6FBA8172EC6D5A505CDAB3927C0A9DE6: 158,
    0x0BC14066C33013FE88F66E314E4CF150B0B2D4D6451A1A51DBBD1C27CD11DE28: 159,
    0x78FDC8D422C49CED035A9EDF18D00D3C6A8D81DF210F3E5E448E045E77B41E88: 160,
    0xAADC37B8BA5645E62F4546802DB221593A94729CCBFC5A97D01365A88F649878: 161,
    0xAAF4F58DE99300CFADC4585755F376D5FA747D5BC561D5BD9D710DE1F91BF42D: 162,
    0x60859188CFFE297F44DDE29F2D2865634621F26215049CAEB304CCBA566A8B17: 163,
    0xE434DC35DA084CF8D7E8186688EA2DACB53DB7003D427AF3ABF351BD9D0A4E8D: 164,
    0xB29A2B3B6F2FF1B765777A231725941DA5072CC4FCC30AC4A2CE09706E8DDEFF: 165,
    0x2DA56674729343ACC9933752C8C469A244252915242EB6D4C02D11DDD69164A1: 166,
    0xB68792697ED876AF8B4858B316F5B54D81F6861191AD2950C1FDE6C3DC7B3DEA: 167,
    0xBEE89403B5BF0E626C2F71ADB366311C697013DF53107181A963ADC459EF4D99: 168,
    0xDC471888E6136F84C49E531E9C9240DC4E3FBA66DA9D3A49E2AF6202133683E0: 169,
    0x550D3DE95BE0BD28A79C3EB4EA7F05692C60B0602E48B49461E703379B08A71A: 170,
    0xFC377260A69A39DD786235C89F4BCD5D9639157731CAC38071A0508750EB115A: 171,
    0x0A0A1BCADD9F6A5539376FA82276E043AE3CB4499DAAAF8136572ECB1F9F0D60: 172,
    0x0440FD76B4E685D17019B0EEF836CEA9994650028B99DDDFB48BE06FA4240AA6: 173,
    0xDF5D400F265039450228FA547DF2BEE79E6A350DAA43FBA4BD328BC654824C64: 174,
    0xDEF993A65205231625280C5E3C23E44B263D0AA948FBC330055626B8AB25A5A1: 175,
    0x238BA8D02078544847438DB7773730A25D584074EAC94489BD8EB86CA267C937: 176,
    0x04CB44C80B6FBF8CEB1D80AF688C9F7C0B2AB5BF4A964CABE37041F23B23F7A8: 177,
    0xBBF265BEA1B905C854054A8DBE97FEDCC06FA54306551423711231A4AD0610C9: 178,
    0x236F2840BFC5DC34B28742DD0B4C9DEFE8A4A5FA9592E49CEFFB9AB51B7EB974: 179,
    0x1C5F5AC147EC2DEE04D8CE29BDBEBBC58F578E0E1392DA66F352A62E5C09C503: 180,
    0x22B88D74A6B23BE687AA96340C881253C2E9873C526EEC7366DC5F733ADA306A: 181,
    0x3AE797CEEF265E3A4F9C1978C47C759EB34A32909251DEE7276DB339B17B3DE3: 182,
    0x6A79CC294E25EB1A13381E9F3361EE96C47EE7ED00BF73ABADB8F9664BFFD0A7: 183,
    0xD91D691C894F8266E3F2D5E558AD2349D6783327A752A4949BC554F514E34988: 184,
    0xE35848A7C6477CFE9366AE64571069FD3A5AD752A460D28C5F73D438B5E432BF: 185,
    0xF3B9EB9E163AF2088B11DE0A369FB583F58F9440E0E5C70FCE0C59909ECECE8A: 186,
    0x28AFDD85196B637A3C64FF1F53AF1AD8DE145CF652297EDE1B38F2CBD6A4B4BF: 187,
    0x6F1F0041084F67CED174808484BD05851DE94443D775585E9D86D4C2589DBA59: 188,
    0xD344F074C815FDED543CD5A29A47659DE529CD0ADB1C1FAE6EDA2D685D422BD8: 189,
    0x4082D8AA0BE13AB143F55D600665A8AE7EF90BA09D57C38FA538A2604D7E9827: 190,
    0xB52CF138A3505DC3D3CD84A77912F4BE1A33DF2C3065D3E4CB37FB1D5D1B5072: 191,
    0x5E29E30C8EA9A89560281B90DBE96FE6F067A8ACC0F164A71449BF0DA7D58D7E: 192,
    0xA4C9B5D989FA12D608052E66DC5A37A431D679E93D0ED25572F97F67460BB157: 193,
    0xB93EDCD1E74716AC76D71E26CE3491BE20745375DCD4848D8F3B91A3F785DBB1: 194,
    0x6D918F650E2B4A9F360977C4447E6376EB632EC1F687BA963AA9983E90086594: 195,
    0x2BDE9B0C0857AEE2CFFDEA6B8723EAF59894499EC278C18F020EDD3C2295E424: 196,
    0xBACDDA17ED986C07F827229709E1DED99D4DA917A5E7E7EC15816EAF2CACF54C: 197,
    0xCFC479828D8133D824A47FE26326D458B6B94134276B945404197F42411564C3: 198,
    0xC1D0558604082AF4380F8AF6E6DF686F24C7438CA4F2A67C86A71EE7852601F9: 199,
    0xE71FAC6FB785942CC6C6404A423F94F32A28AE66D69FF41494C38BFD4788B2F8: 200,
    0x66BE4F155C5EF2EBD3772B228F2F00681E4ED5826CDB3B1943CC11AD15AD1D28: 201,
    0x42D72674974F694B5F5159593243114D38A5C39C89D6B62FEE061FF523240EE1: 202,
    0xA7CE836D032B2BF62B7E2097A8E0A6D8AEB35405AD15271E96D3B0188A1D06FB: 203,
    0x47197230E1E4B29FC0BD84D7D78966C0925452AFF72A2A121538B102457E9EBE: 204,
    0x83978B4C69C48DD978AB43FE30F077615294F938FB7F936D9EB340E51EA7DB2E: 205,
    0xD36CD1C74EF8D7326D8021B776C18FB5A5724B7F7BC93C2F42E43E10EF27D12A: 206,
    0xACB8D954E2CFEF495862221E91BD7523613CF8808827CB33EDFE4904CC51BF29: 207,
    0xE89D44C8FD6A9BAC8AF33CE47F56337617D449BF7FF3956B618C646DE829CBCB: 208,
    0x695FB3134AD82C3B8022BC5464EDD0BCC9424EF672B52245DCB6AB2374327CE3: 209,
    0xF2192E1030363415D7B4FB0406540A0060E8E2FC8982F3F32289379E11FA6546: 210,
    0x915C3EB987B20E1AF620C1403197BF687FB7F18513B3A73FDE6E78C7072C41A6: 211,
    0x9780E26D96B1F2A9A18EF8FC72D589DBF03EF788137B64F43897E83A91E7FEEC: 212,
    0x51858DE9989BF7441865EBDADBF7382C8838EDBF830F5D86A9A51AC773676DD6: 213,
    0xE767803F8ECF1DEE6BB0345811F7312CDA556058B19DB6389AD9AE3568643DDD: 214,
    0x8A012A6DE2943A5AA4D77ACF5E695D4456760A3F1F30A5D6DC2079599187A071: 215,
    0x5320AD99A619A90804CD2EFE3A5CF0AC1AC5C41AD9FF2C61CF699EFDAD771096: 216,
    0xCC6782FD46DD71C5F512301AB049782450B4EAF79FDAC5443D93D274D3916786: 217,
    0xB3D6E86317C38844915B053A0C35FF2FC103B684E96CEF2918AB06844EB51AAF: 218,
    0x4C0D3471EAD8EE99FBD8249E33F683E07C6CD6071FE102DD09617B2C353DE430: 219,
    0x3162B0988D4210BFF484413ED451D170A03887272177EFC0B7D000F10ABE9EDF: 220,
    0xAC507B9F8BF86AD8BB770F71CD2B1992902AE0314D93FC0F2BB011D70E796226: 221,
    0xFAE8130C0619F84B4B44F01B84806F04E82E536D70E05F2356977FA318AECC1A: 222,
    0x65E3D48FA860A761B461CE1274F0D562F3DB9A6A57CF04D8C90D68F5670B6AEA: 223,
    0x8B43726243EEAF8325404568ABECE3264B546CF9D88671F09C24C87045FCCB4F: 224,
    0x3EFDD7A884FF9E18C9E5711C185AA6C5E413B68F23197997DA5B1665CA978F99: 225,
    0x26A62D79192C78C3891F38189368673110B88734C09ED7453515DEF7525E07D8: 226,
    0x37F6A7F96B945F2F9A9127CCB4A8552FCB6938E53FE8F046DB8DA238398093E9: 227,
    0x04E4A0BB093261EE16386DADCEF9E2A83913F4E1899464891421D20C1BBFF74D: 228,
    0x5625F7C930B8B40DE87DC8E69145D83FD1D81C61B6C31FB7CFE69FAC65B28642: 229,
    0xD31DDB47B5E8664717D3718ACBD132396FF496FE337159C99410BE8658408A27: 230,
    0x6CB0DB1D7354DFB4A1464318006DF0643CAFE2002A86A29FF8560F900FEF28A1: 231,
    0x53C8DA29BFA275271DF3F270296D5A7D61B57F8848C89B3F65F49E21340B7592: 232,
    0xEA6426B4B8D70CAA8ECE9A88FB0A9D4A6B817BB4A43AC6FBEF64CB0E589129EE: 233,
    0x61C831BEAB28D67D1BB40B5AE1A11E2757FA842F031A2D0BC94A7867BC5D26C2: 234,
    0x0446C598F3355ED7D8A3B7E0B99F9299D15E956A97FAAE081A0B49D17024ABD2: 235,
    0xE7DFAC380F4A6ED3A03E62F813161EFF828766FA014393558E075E9CEB77D549: 236,
    0x0504E0A132D2EF5CA5F2FE74FC64437205BC10F32D5F13D533BF552916A94D3F: 237,
    0xDB444DA68C84F0A9CE08609100B69B8F3D5672687E0CA13FA3C0AC9EB2BDE5D2: 238,
    0xDD0DC620E7584674CB3DBA490D2EBA9E68ECA0BEF228EE569A4A64F6559056E9: 239,
    0x681483E2251CD5E2885507BB09F76BED3B99D3C377DD48396177647BFB4AAFDA: 240,
    0xC29B39917E4E60F0FEE5B6871B30A38E50531D76D1B0837811BD6351B34854EC: 241,
    0x83D76AFC3887C0B7EDD14A1AFFA7554BED3345BA68DDCD2A3326C7EAE97B80D8: 242,
    0x2F5553803273E8BB29D913CC31BAB953051C59F3BA57A71CF5591563CA721405: 243,
    0xFC6A672327474E1387FCBCE1814A1DE376D8561FC138561441AC6E396089E062: 244,
    0x81630654DFB0FD282A37117995646CDDE2CF8EEFE9F3F96FDB12CFDA88DF6668: 245,
    0xDDF78CFA378B5E068A248EDAF3ABEF23EA9E62C66F86F18CC5E695CD36C9809B: 246,
    0xE9944EBEF6E5A24035A31A727E8FF6DA7C372D99949C1224483B857F6401E346: 247,
    0x6120B123382F98F7EFE66ABE6A3A3445788A87E48D4E6991F37BAADCAC0BEF95: 248,
    0x168C8166292B85070409830617E84BDD7E3518B38E5AC430DC35ED7D16B07A86: 249,
    0xD84F57F3FFA76CC18982DA4353CC5991158EC5AE4F6A9109D1D7A0AE2CBA77ED: 250,
    0x3E7257B7272BB46D49CD6019B04DDEE20DA7C0CB13F7C1EC3391291B2CCEBABC: 251,
    0x371F36870D18F32A11FEA0F144B021C8B407BB50F8E0267C711123F454B963C0: 252,
    0x9346AC6DD7DE6B96975FEC380D4D994C4C12E6A8897544F22915316CC6CCA280: 253,
    0x54075DF80EC1AE6AC9100E1FD0EBF3246C17F5C933137AF392011F4C5F61513A: 254,
    0xE08EC2AF2CFC251225E1968FD6CA21E4044F129BFFA95BAC3503BE8BDB30A367: 255,
}

sha3_inv_offset: dict[int, tuple[int, int]] = mk_sha3_inv_offset(sha3_inv)


class NamedTimer:
    def __init__(self, name: str, auto_start=True):
        self.name = name
        self.start_time = timer() if auto_start else None
        self.end_time = None
        self.sub_timers = []

    def start(self):
        if self.start_time is not None:
            raise ValueError(f"Timer {self.name} has already been started.")
        self.start_time = timer()

    def stop(self, stop_subtimers=True):
        if stop_subtimers:
            for sub_timer in self.sub_timers:
                sub_timer.stop()

        # if the timer has already been stopped, do nothing
        self.end_time = self.end_time or timer()

    def create_subtimer(self, name, auto_start=True, stop_previous=True):
        for subtimer in self.sub_timers:
            if subtimer.name == name:
                raise ValueError(f"Timer with name {name} already exists.")

        if stop_previous and self.sub_timers:
            self.sub_timers[-1].stop()

        sub_timer = NamedTimer(name, auto_start=auto_start)
        self.sub_timers.append(sub_timer)
        return sub_timer

    def __getitem__(self, name):
        for subtimer in self.sub_timers:
            if subtimer.name == name:
                return subtimer
        raise ValueError(f"Timer with name {name} does not exist.")

    def elapsed(self) -> float:
        if self.start_time is None:
            raise ValueError(f"Timer {self.name} has not been started")

        end_time = self.end_time if self.end_time is not None else timer()

        return end_time - self.start_time

    def report(self, include_subtimers=True) -> str:
        sub_reports_str = ""

        if include_subtimers:
            sub_reports = [
                f"{timer.name}: {timer.elapsed():.2f}s" for timer in self.sub_timers
            ]
            sub_reports_str = f" ({', '.join(sub_reports)})" if sub_reports else ""

        return f"{self.name}: {self.elapsed():.2f}s{sub_reports_str}"

    def __str__(self):
        return self.report()

    def __repr__(self):
        return (
            f"NamedTimer(name={self.name}, start_time={self.start_time}, "
            f"end_time={self.end_time}, sub_timers={self.sub_timers})"
        )


def format_size(num_bytes: int) -> str:
    """
    Returns a human-readable string for a number of bytes
    Automatically chooses a relevant size unit (G, M, K, B)

    e.g.:
        1234567890 -> 1.15G
        123456789 -> 117.7M
        123456 -> 120.5K
        123 -> 123B
    """
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f}GB"
    elif num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f}MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f}KB"
    else:
        return f"{num_bytes}B"


def format_time(seconds: float) -> str:
    """
    Returns a pretty string for an elapsed time in seconds.
    Automatically chooses a relevant time unit (h, m, s, ms, µs, ns)

    Examples:
        3602.13 -> 1h00m02s
        62.003 -> 1m02s
        1.000000001 -> 1.000s
        0.123456789 -> 123.457ms
        0.000000001 -> 1.000ns
    """
    if seconds >= 3600:
        # 1 hour or more
        hours = int(seconds / 3600)
        minutes = int((seconds - (3600 * hours)) / 60)
        seconds_rounded = int(seconds - (3600 * hours) - (60 * minutes))
        return f"{hours}h{minutes:02}m{seconds_rounded:02}s"
    elif seconds >= 60:
        # 1 minute or more
        minutes = int(seconds / 60)
        seconds_rounded = int(seconds - (60 * minutes))
        return f"{minutes}m{seconds_rounded:02}s"
    elif seconds >= 1:
        # 1 second or more
        return f"{seconds:.3f}s"
    elif seconds >= 1e-3:
        # 1 millisecond or more
        return f"{seconds * 1e3:.3f}ms"
    elif seconds >= 1e-6:
        # 1 microsecond or more
        return f"{seconds * 1e6:.3f}µs"
    else:
        # Otherwise, display in nanoseconds
        return f"{seconds * 1e9:.3f}ns"


def parse_time(arg: int | float | str, default_unit: str | None = "s") -> float:
    """
    Parse a time string into a number of seconds, with an optional unit suffix.

    Examples:
        "200ms" -> 0.2
        "5s" -> 5.0
        "2m" -> 120.0
        "1h" -> 3600.0

    Note: does not support combined units like "1h00m02s" (like `format_time` produces)
    """

    if default_unit and default_unit not in ["ms", "s", "m", "h"]:
        raise ValueError(f"Invalid time unit: {default_unit}")

    if isinstance(arg, str):
        if arg.endswith("ms"):
            return float(arg[:-2]) / 1000
        elif arg.endswith("s"):
            return float(arg[:-1])
        elif arg.endswith("m"):
            return float(arg[:-1]) * 60
        elif arg.endswith("h"):
            return float(arg[:-1]) * 3600
        elif arg == "0":
            return 0.0
        else:
            if not default_unit:
                raise ValueError(f"Could not infer time unit from {arg}")
            return parse_time(arg + default_unit, default_unit=None)
    elif isinstance(arg, int | float):
        if not default_unit:
            raise ValueError(f"Could not infer time unit from {arg}")
        return parse_time(str(arg) + default_unit, default_unit=None)
    else:
        raise ValueError(f"Invalid time argument: {arg}")


class timed_block:
    def __init__(self, label="Block"):
        self.label = label

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = timer()
        elapsed = end - self.start
        print(f"{self.label} took {format_time(elapsed)}")


def timed(label=None):
    """
    A decorator that measures and prints the execution time of a function.

    Args:
        label (str, optional): Custom label for the timing output. If None, the function name will be used.

    Returns:
        callable: The wrapped function with timing functionality.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use function name if no label is provided
            function_label = label if label is not None else func.__name__

            with timed_block(function_label):
                result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator
