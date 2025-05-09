# SPDX-License-Identifier: AGPL-3.0

import json
import re
from dataclasses import dataclass
from subprocess import PIPE, Popen

from xxhash import xxh3_64, xxh3_64_digest
from z3 import (
    ULT,
    And,
    BitVec,
    BitVecRef,
    Concat,
    Function,
    Implies,
    Not,
    Or,
    eq,
    is_bv,
    is_bv_value,
    is_false,
    simplify,
    unsat,
)

from .assertions import assert_cheatcode_handler
from .bytevec import ByteVec
from .calldata import (
    FunctionInfo,
    get_abi,
    mk_calldata,
)
from .exceptions import FailCheatcode, HalmosException, InfeasiblePath, NotConcreteError
from .logs import debug
from .mapper import BuildOut
from .utils import (
    Address,
    BitVecSort8,
    BitVecSort160,
    BitVecSort256,
    BitVecSorts,
    Word,
    assert_address,
    con,
    con_addr,
    decode_hex,
    extract_bytes,
    extract_funsig,
    extract_string_argument,
    f_ecrecover,
    green,
    hexify,
    int256,
    int_of,
    is_non_zero,
    red,
    secp256k1n,
    stripped,
    uid,
    uint160,
    uint256,
)

# f_vmaddr(key) -> address
f_vmaddr = Function("f_vmaddr", BitVecSort256, BitVecSort160)

# f_sign_v(key, digest) -> v
f_sign_v = Function("f_sign_v", BitVecSort256, BitVecSort256, BitVecSort8)

# f_sign_r(key, digest) -> r
f_sign_r = Function("f_sign_r", BitVecSort256, BitVecSort256, BitVecSort256)

# f_sign_s(key, digest) -> s
f_sign_s = Function("f_sign_s", BitVecSort256, BitVecSort256, BitVecSort256)


def name_of(x: str) -> str:
    if not isinstance(x, str):
        raise NotConcreteError(f"expected concrete string but got: {x}")

    return re.sub(r"\s+", "_", x)


def extract_string_array_argument(calldata: BitVecRef, arg_idx: int):
    """Extracts idx-th argument of string array from calldata"""

    array_slot = int_of(extract_bytes(calldata, 4 + 32 * arg_idx, 32))
    num_strings = int_of(extract_bytes(calldata, 4 + array_slot, 32))

    string_array = []

    for i in range(num_strings):
        string_offset = int_of(
            extract_bytes(calldata, 4 + array_slot + 32 * (i + 1), 32)
        )
        string_length = int_of(
            extract_bytes(calldata, 4 + array_slot + 32 + string_offset, 32)
        )
        string_value = int_of(
            extract_bytes(
                calldata, 4 + array_slot + 32 + string_offset + 32, string_length
            )
        )
        string_bytes = string_value.to_bytes(string_length, "big")
        string_array.append(string_bytes.decode("utf-8"))

    return string_array


def stringified_bytes_to_bytes(hexstring: str) -> ByteVec:
    """Converts a string of bytes to a bytes memory type"""

    hexstring = stripped(hexstring)
    hexstring_len = (len(hexstring) + 1) // 2
    hexstring_len_enc = stripped(hex(hexstring_len)).rjust(64, "0")
    hexstring_len_ceil = (hexstring_len + 31) // 32 * 32

    ret_bytes = bytes.fromhex(
        "00" * 31
        + "20"
        + hexstring_len_enc
        + hexstring.ljust(hexstring_len_ceil * 2, "0")
    )

    return ByteVec(ret_bytes)


@dataclass(frozen=True)
class PrankResult:
    sender: Address | None = None
    origin: Address | None = None

    def __bool__(self) -> bool:
        """
        True iff either sender or origin is set.
        """
        return self.sender is not None or self.origin is not None

    def __str__(self) -> str:
        return f"{hexify(self.sender)}, {hexify(self.origin)}"


NO_PRANK = PrankResult()


@dataclass
class Prank:
    """
    A mutable object to store the current prank context.

    Because it's mutable, it must be copied across contexts.

    Can test for the existence of an active prank with `if prank: ...`

    A prank is active if either sender or origin is set.

    - prank(address) sets sender
    - prank(address, address) sets both sender and origin
    """

    active: PrankResult = NO_PRANK  # active prank context
    keep: bool = False  # start / stop prank

    def __bool__(self) -> bool:
        """
        True iff either sender or origin is set.
        """
        return bool(self.active)

    def __str__(self) -> str:
        if not self:
            return "no active prank"

        fn_name = "startPrank" if self.keep else "prank"
        return f"{fn_name}({str(self.active)})"

    def lookup(self, to: Address) -> PrankResult:
        """
        If `to` is an eligible prank destination, return the active prank context.

        If `keep` is False, this resets the prank context.
        """

        assert_address(to)
        if (
            self
            and not eq(to, hevm_cheat_code.address)
            and not eq(to, halmos_cheat_code.address)
        ):
            result = self.active
            if not self.keep:
                self.stopPrank()
            return result

        return NO_PRANK

    def prank(
        self, sender: Address, origin: Address | None = None, _keep: bool = False
    ) -> bool:
        assert_address(sender)
        if self.active:
            return False

        self.active = PrankResult(sender=sender, origin=origin)
        self.keep = _keep
        return True

    def startPrank(self, sender: Address, origin: Address | None = None) -> bool:
        return self.prank(sender, origin, _keep=True)

    def stopPrank(self) -> bool:
        # stopPrank calls are allowed even when no active prank exists
        self.active = NO_PRANK
        self.keep = False
        return True


def symbolic_storage(ex, arg, sevm, stack, step_id):
    account = uint160(arg.get_word(4))
    account_alias = sevm.resolve_address_alias(
        ex, account, stack, step_id, allow_branching=False
    )

    if account_alias is None:
        error_msg = f"enableSymbolicStorage() is not allowed for a nonexistent account: {hexify(account)}"
        raise HalmosException(error_msg)

    ex.storage[account_alias].symbolic = True

    return ByteVec()  # empty return data


def snapshot_storage(ex, arg, sevm, stack, step_id):
    account = uint160(arg.get_word(4))
    account_alias = sevm.resolve_address_alias(
        ex, account, stack, step_id, allow_branching=False
    )

    if account_alias is None:
        error_msg = f"snapshotStorage() is not allowed for a nonexistent account: {hexify(account)}"
        raise HalmosException(error_msg)

    zero_pad = b"\x00" * 16
    return ByteVec(zero_pad + ex.storage[account_alias].digest())


def snapshot_state(
    ex, arg=None, sevm=None, stack=None, step_id=None, include_path=False
):
    """
    Generates a snapshot ID by hashing the current state (balance, code, and storage), including constraints over state variables if include_path is set.

    The snapshot ID is constructed by concatenating four hashes: balance (64 bits), code (64 bits), storage (64 bits), and constraints (64 bits).
    """
    # balance
    balance_hash = xxh3_64_digest(int.to_bytes(ex.balance.get_id(), length=32))

    # code
    m = xxh3_64()
    # note: iteration order is guaranteed to be the insertion order
    for addr, code in ex.code.items():
        m.update(int.to_bytes(int_of(addr), length=32))
        # simply the object address is used, as code remains unchanged after deployment
        m.update(int.to_bytes(id(code), length=32))
    code_hash = m.digest()

    # storage
    m = xxh3_64()
    for addr, storage in ex.storage.items():
        m.update(int.to_bytes(int_of(addr), length=32))
        m.update(storage.digest())
    storage_hash = m.digest()

    # path
    m = xxh3_64()
    if include_path:
        if ex.path.sliced is None:
            raise ValueError("path not yet sliced")
        for idx, cond in enumerate(ex.path.conditions):
            if idx in ex.path.sliced:
                m.update(int.to_bytes(cond.get_id(), length=32))
    path_hash = m.digest()

    return ByteVec(balance_hash + code_hash + storage_hash + path_hash)


def create_calldata_contract(ex, arg, sevm, stack, step_id):
    contract_name = name_of(extract_string_argument(arg, 0))
    return create_calldata_generic(ex, sevm, contract_name)


def create_calldata_contract_bool(ex, arg, sevm, stack, step_id):
    contract_name = name_of(extract_string_argument(arg, 0))
    include_view = int_of(
        extract_bytes(arg, 4 + 32 * 1, 32),
        "symbolic boolean flag for SVM.createCalldata()",
    )
    return create_calldata_generic(
        ex, sevm, contract_name, include_view=bool(include_view)
    )


def create_calldata_file_contract(ex, arg, sevm, stack, step_id):
    filename = name_of(extract_string_argument(arg, 0))
    contract_name = name_of(extract_string_argument(arg, 1))
    return create_calldata_generic(ex, sevm, contract_name, filename)


def create_calldata_file_contract_bool(ex, arg, sevm, stack, step_id):
    filename = name_of(extract_string_argument(arg, 0))
    contract_name = name_of(extract_string_argument(arg, 1))
    include_view = int_of(
        extract_bytes(arg, 4 + 32 * 2, 32),
        "symbolic boolean flag for SVM.createCalldata()",
    )
    return create_calldata_generic(
        ex, sevm, contract_name, filename, bool(include_view)
    )


def encode_tuple_bytes(data: BitVecRef | ByteVec | bytes) -> ByteVec:
    """
    Return ABI encoding of a tuple containing a single bytes element.

    encoding of a tuple (bytes): 32 (offset) + length + data
    """

    length = data.size() // 8 if is_bv(data) else len(data)
    result = ByteVec((32).to_bytes(32) + int(length).to_bytes(32))
    result.append(data)
    return result


def create_calldata_generic(ex, sevm, contract_name, filename=None, include_view=False):
    """
    Generate arbitrary symbolic calldata for the given contract.

    Dynamic-array arguments are sized in the same way of regular test functions.

    The contract is identified by its contract name and optional filename.
    TODO: provide variants that require only the contract address.
    """
    contract_json = BuildOut().get_by_name(contract_name, filename)

    abi = get_abi(contract_json)
    methodIdentifiers = contract_json["methodIdentifiers"]

    results = []

    # empty calldata for receive() and fallback()
    results.append(encode_tuple_bytes(b""))

    # nonempty calldata for fallback()
    fallback_selector = BitVec(
        f"fallback_selector_{uid()}_{ex.new_symbol_id():>02}", 4 * 8
    )
    fallback_input_length = 1024  # TODO: configurable
    fallback_input = BitVec(
        f"fallback_input_{uid()}_{ex.new_symbol_id():>02}", fallback_input_length * 8
    )
    results.append(encode_tuple_bytes(Concat(fallback_selector, fallback_input)))

    for funsig in methodIdentifiers:
        funname = funsig.split("(")[0]
        funselector = methodIdentifiers[funsig]
        funinfo = FunctionInfo(funname, funsig, funselector)

        # assume fallback_selector differs from all existing selectors
        ex.path.append(fallback_selector != con(int(funselector, 16), 32))

        if not include_view:
            fun_abi = abi[funinfo.sig]
            if fun_abi["stateMutability"] in ["pure", "view"]:
                continue

        calldata, dyn_params = mk_calldata(
            abi,
            funinfo,
            sevm.options,
            new_symbol_id=ex.new_symbol_id,
        )
        # TODO: this may accumulate dynamic size candidates from multiple calldata into a single path object,
        # which is not optimal, as unnecessary size candidates will need to be copied during path branching for each calldata.
        ex.path.process_dyn_params(dyn_params)

        results.append(encode_tuple_bytes(calldata))

    return results


def create_generic(ex, bits: int, var_name: str, type_name: str) -> BitVecRef:
    label = f"halmos_{var_name}_{type_name}_{uid()}_{ex.new_symbol_id():>02}"
    return BitVec(label, BitVecSorts[bits])


def create_uint(ex, arg, **kwargs):
    bits = int_of(
        extract_bytes(arg, 4, 32), "symbolic bit size for halmos.createUint()"
    )
    if bits > 256:
        raise HalmosException(f"bitsize larger than 256: {bits}")

    name = name_of(extract_string_argument(arg, 1))
    return ByteVec(uint256(create_generic(ex, bits, name, f"uint{bits}")))


def create_uint256(ex, arg, **kwargs):
    name = name_of(extract_string_argument(arg, 0))
    return ByteVec(create_generic(ex, 256, name, "uint256"))


def create_int(ex, arg, **kwargs):
    bits = int_of(
        extract_bytes(arg, 4, 32), "symbolic bit size for halmos.createUint()"
    )
    if bits > 256:
        raise HalmosException(f"bitsize larger than 256: {bits}")

    name = name_of(extract_string_argument(arg, 1))
    return ByteVec(int256(create_generic(ex, bits, name, f"int{bits}")))


def create_int256(ex, arg, **kwargs):
    name = name_of(extract_string_argument(arg, 0))
    return ByteVec(create_generic(ex, 256, name, "int256"))


def create_bytes(ex, arg, **kwargs):
    byte_size = int_of(
        extract_bytes(arg, 4, 32), "symbolic byte size for halmos.createBytes()"
    )
    name = name_of(extract_string_argument(arg, 1))
    symbolic_bytes = create_generic(ex, byte_size * 8, name, "bytes")
    return encode_tuple_bytes(symbolic_bytes)


def create_string(ex, arg, **kwargs):
    byte_size = int_of(
        extract_bytes(arg, 4, 32), "symbolic byte size for halmos.createString()"
    )
    name = name_of(extract_string_argument(arg, 1))
    symbolic_string = create_generic(ex, byte_size * 8, name, "string")
    return encode_tuple_bytes(symbolic_string)


def create_bytes4(ex, arg, **kwargs):
    name = name_of(extract_string_argument(arg, 0))
    result = ByteVec(create_generic(ex, 32, name, "bytes4"))
    result.append((0).to_bytes(28))  # pad right
    return result


def create_bytes32(ex, arg, **kwargs):
    name = name_of(extract_string_argument(arg, 0))
    return ByteVec(create_generic(ex, 256, name, "bytes32"))


def create_address(ex, arg, **kwargs):
    name = name_of(extract_string_argument(arg, 0))
    return ByteVec(uint256(create_generic(ex, 160, name, "address")))


def create_bool(ex, arg, **kwargs):
    name = name_of(extract_string_argument(arg, 0))
    return ByteVec(uint256(create_generic(ex, 1, name, "bool")))


def apply_vmaddr(ex, private_key: Word):
    # check if this private key has an existing address associated with it
    known_keys = ex.known_keys
    addr = known_keys.get(private_key, None)
    if addr is None:
        # if not, create a new address
        addr = f_vmaddr(private_key)

        # mark the addresses as distinct
        for other_key, other_addr in known_keys.items():
            distinct = Implies(private_key != other_key, addr != other_addr)
            ex.path.append(distinct)

        # associate the new address with the private key
        known_keys[private_key] = addr

    return addr


class halmos_cheat_code:
    # address constant SVM_ADDRESS =
    #     address(bytes20(uint160(uint256(keccak256('svm cheat code')))));
    address: BitVecRef = con_addr(0xF3993A62377BCD56AE39D773740A5390411E8BC9)

    handlers = {
        0x66830DFA: create_uint,  # createUint(uint256,string)
        0xBC7BEEFC: create_uint256,  # createUint256(string)
        0x49B9C7D4: create_int,  # createInt(uint256,string)
        0xC2CE6AED: create_int256,  # createInt256(string)
        0xEEF5311D: create_bytes,  # createBytes(uint256,string)
        0xCE68656C: create_string,  # createString(uint256,string)
        0xDE143925: create_bytes4,  # createBytes4(string)
        0xBF72FA66: create_bytes32,  # createBytes32(string)
        0x3B0FA01B: create_address,  # createAddress(string)
        0x6E0BB659: create_bool,  # createBool(string)
        0xDC00BA4D: symbolic_storage,  # enableSymbolicStorage(address)
        0x5DBB8438: snapshot_storage,  # snapshotStorage(address)
        0x9CD23835: snapshot_state,  # snapshotState()
        0xBE92D5A2: create_calldata_contract,  # createCalldata(string)
        0xDEEF391B: create_calldata_contract_bool,  # createCalldata(string,bool)
        0x88298B32: create_calldata_file_contract,  # createCalldata(string,string)
        0x607C5C90: create_calldata_file_contract_bool,  # createCalldata(string,string,bool)
    }

    @staticmethod
    def handle(sevm, ex, arg: BitVecRef, stack, step_id) -> list[BitVecRef]:
        funsig = int_of(extract_funsig(arg), "symbolic halmos cheatcode")
        if handler := halmos_cheat_code.handlers.get(funsig):
            result = handler(ex, arg, sevm=sevm, stack=stack, step_id=step_id)
            return result if isinstance(result, list) else [result]

        error_msg = f"Unknown halmos cheat code: function selector = 0x{funsig:0>8x}, calldata = {hexify(arg)}"
        raise HalmosException(error_msg)


class hevm_cheat_code:
    # https://github.com/dapphub/ds-test/blob/cd98eff28324bfac652e63a239a60632a761790b/src/test.sol

    # address constant HEVM_ADDRESS =
    #     address(bytes20(uint160(uint256(keccak256('hevm cheat code')))));
    address: BitVecRef = con_addr(0x7109709ECFA91A80626FF3989D68F67F5B1DD12D)

    # abi.encodePacked(
    #     bytes4(keccak256("store(address,bytes32,bytes32)")),
    #     abi.encode(HEVM_ADDRESS, bytes32("failed"), bytes32(uint256(0x01)))
    # )
    fail_payload = ByteVec(
        bytes.fromhex(
            "70ca10bb"
            + "0000000000000000000000007109709ecfa91a80626ff3989d68f67f5b1dd12d"
            + "6661696c65640000000000000000000000000000000000000000000000000000"
            + "0000000000000000000000000000000000000000000000000000000000000001"
        )
    )

    dict_of_unsupported_cheatcodes = {
        0xBD6AF434: "expectCall(address,bytes)",
        0xC1ADBBFF: "expectCall(address,bytes,uint64)",
        0xF30C7BA3: "expectCall(address,uint256,bytes)",
        0xA2B1A1AE: "expectCall(address,uint256,bytes,uint64)",
        0x440ED10D: "expectEmit()",
        0x86B9620D: "expectEmit(address)",
        0x491CC7C2: "expectEmit(bool,bool,bool,bool)",
        0x81BAD6F3: "expectEmit(bool,bool,bool,bool,address)",
        0x11FB5B9C: "expectPartialRevert(bytes4)",
        0x51AA008A: "expectPartialRevert(bytes4,address)",
        0xF4844814: "expectRevert()",
        0xD814F38A: "expectRevert(address)",
        0x1FF5F952: "expectRevert(address,uint64)",
        0xF28DCEB3: "expectRevert(bytes)",
        0x61EBCF12: "expectRevert(bytes,address)",
        0xD345FB1F: "expectRevert(bytes,address,uint64)",
        0x4994C273: "expectRevert(bytes,uint64)",
        0xC31EB0E0: "expectRevert(bytes4)",
        0x260BC5DE: "expectRevert(bytes4,address)",
        0xB0762D73: "expectRevert(bytes4,address,uint64)",
        0xE45CA72D: "expectRevert(bytes4,uint64)",
        0x4EE38244: "expectRevert(uint64)",
        0x65BC9481: "accesses(address)",
        0xAFC98040: "broadcast()",
        0xE6962CDB: "broadcast(address)",
        0xF67A965B: "broadcast(uint256)",
        0x3FDF4E15: "clearMockedCalls()",
        0x08D6B37A: "deleteStateSnapshot(uint256)",
        0xE0933C74: "deleteStateSnapshots()",
        0x796B89B9: "getBlockTimestamp()",
        0xA5748AAD: "getNonce((address,uint256,uint256,uint256))",
        0x2D0335AB: "getNonce(address)",
        0x191553A4: "getRecordedLogs()",
        0x64AF255D: "isContext(uint8)",
        0xB96213E4: "mockCall(address,bytes,bytes)",
        0x81409B91: "mockCall(address,uint256,bytes,bytes)",
        0xDBAAD147: "mockCallRevert(address,bytes,bytes)",
        0xD23CD037: "mockCallRevert(address,uint256,bytes,bytes)",
        0x5C5C3DE9: "mockCalls(address,bytes,bytes[])",
        0x08BCBAE1: "mockCalls(address,uint256,bytes,bytes[])",
        0xADF84D21: "mockFunction(address,address,bytes)",
        0xD1A5B36F: "pauseGasMetering()",
        0x7D73D042: "prank(address,address,bool)",
        0xA7F8BF5C: "prank(address,bool)",
        0x3B925549: "prevrandao(bytes32)",
        0x4AD0BAC9: "readCallers()",
        0x266CF109: "record()",
        0x41AF2F52: "recordLogs()",
        0xBE367DD3: "resetGasMetering()",
        0x2BCD50E0: "resumeGasMetering()",
        0xC2527405: "revertToState(uint256)",
        0x3A1985DC: "revertToStateAndDelete(uint256)",
        0xF8E18B57: "setNonce(address,uint64)",
        0xDD9FCA12: "snapshotGasLastCall(string)",
        0x200C6772: "snapshotGasLastCall(string,string)",
        0x6D2B27D8: "snapshotValue(string,string,uint256)",
        0x51DB805A: "snapshotValue(string,uint256)",
        0x7FB5297F: "startBroadcast()",
        0x7FEC2A8D: "startBroadcast(address)",
        0xCE817D47: "startBroadcast(uint256)",
        0x4EB859B5: "startPrank(address,address,bool)",
        0x1CC0B435: "startPrank(address,bool)",
        0x3CAD9D7B: "startSnapshotGas(string)",
        0x6CD0CC53: "startSnapshotGas(string,string)",
        0xCF22E3C9: "startStateDiffRecording()",
        0xAA5CF90E: "stopAndReturnStateDiff()",
        0x76EADD36: "stopBroadcast()",
        0xF6402EDA: "stopSnapshotGas()",
        0x773B2805: "stopSnapshotGas(string)",
        0x0C9DB707: "stopSnapshotGas(string,string)",
        0x48F50C0F: "txGasPrice(uint256)",
        0x285B366A: "assumeNoRevert()",
        0x98680034: "createSelectFork(string)",
        0x2F103F22: "activeFork()",
        0xEA060291: "allowCheatcodes(address)",
        0x31BA3498: "createFork(string)",
        0x7CA29682: "createFork(string,bytes32)",
        0x6BA3BA2B: "createFork(string,uint256)",
        0x84D52B7A: "createSelectFork(string,bytes32)",
        0x71EE464D: "createSelectFork(string,uint256)",
        0xD92D8EFD: "isPersistent(address)",
        0x57E22DDE: "makePersistent(address)",
        0x4074E0A8: "makePersistent(address,address)",
        0xEFB77A75: "makePersistent(address,address,address)",
        0x1D9E269E: "makePersistent(address[])",
        0x997A0222: "revokePersistent(address)",
        0x3CE969E6: "revokePersistent(address[])",
        0x0F29772B: "rollFork(bytes32)",
        0xD9BBF3A1: "rollFork(uint256)",
        0xF2830F7B: "rollFork(uint256,bytes32)",
        0xD74C83A4: "rollFork(uint256,uint256)",
        0x9EBF6827: "selectFork(uint256)",
        0xBE646DA1: "transact(bytes32)",
        0x4D8ABC4B: "transact(uint256,bytes32)",
        0x42181150: "envInt(string,string)",
        0x74318528: "envOr(string,string,uint256[])",
        0x97949042: "envBytes32(string)",
        0x350D56BF: "envAddress(string)",
        0xAD31B9FA: "envAddress(string,string)",
        0x7ED1EC7D: "envBool(string)",
        0xAAADDEAF: "envBool(string,string)",
        0x953C097E: "envBytes(bytes)",
        0x6C42F03F: "envBytes(bytes,bytes)",
        0x5AF231C1: "envBytes32(string,string)",
        0x892A0C61: "envInt(string)",
        0x561FE540: "envOr(string,address)",
        0x4777F3CF: "envOr(string,bool)",
        0xB3E47705: "envOr(string,bytes)",
        0xB4A85892: "envOr(string,bytes32)",
        0xBBCB713E: "envOr(string,int256)",
        0xD145736C: "envOr(string,string)",
        0xC74E9DEB: "envOr(string,string,address[])",
        0xEB85E83B: "envOr(string,string,bool[])",
        0x2281F367: "envOr(string,string,bytes32[])",
        0x64BC3E64: "envOr(string,string,bytes[])",
        0x4700D74B: "envOr(string,string,int256[])",
        0x859216BC: "envOr(string,string,string[])",
        0x5E97348F: "envOr(string,uint256)",
        0xF877CB19: "envString(string)",
        0x14B02BC9: "envString(string,string)",
        0xC1978D1F: "envUint(string)",
        0xF3DEC099: "envUint(string,string)",
        0x3EBF73B4: "getDeployedCode(string)",
        0x528A683C: "keyExists(string,string)",
        0xDB4235F6: "keyExistsJson(string,string)",
        0x600903AD: "keyExistsToml(string,string)",
        0x6A82600A: "parseJson(string)",
        0x85940EF1: "parseJson(string,string)",
        0x213E4198: "parseJsonKeys(string,string)",
        0x592151F0: "parseToml(string)",
        0x37736E08: "parseToml(string,string)",
        0x812A44B2: "parseTomlKeys(string,string)",
        0xD930A0E6: "projectRoot()",
        0x47EAF474: "prompt(string)",
        0x1E279D41: "promptSecret(string)",
        0x69CA02B7: "promptSecretUint(string)",
        0x972C6062: "serializeAddress(string,string,address)",
        0x1E356E1A: "serializeAddress(string,string,address[])",
        0xAC22E971: "serializeBool(string,string,bool)",
        0x92925AA1: "serializeBool(string,string,bool[])",
        0xF21D52C7: "serializeBytes(string,string,bytes)",
        0x9884B232: "serializeBytes(string,string,bytes[])",
        0x2D812B44: "serializeBytes32(string,string,bytes32)",
        0x201E43E2: "serializeBytes32(string,string,bytes32[])",
        0x3F33DB60: "serializeInt(string,string,int256)",
        0x7676E127: "serializeInt(string,string,int256[])",
        0x9B3358B0: "serializeJson(string,string)",
        0x88DA6D35: "serializeString(string,string,string)",
        0x561CD6F3: "serializeString(string,string,string[])",
        0x129E9002: "serializeUint(string,string,uint256)",
        0xFEE9A469: "serializeUint(string,string,uint256[])",
        0x3D5923EE: "setEnv(string,string)",
        0xFA9D8713: "sleep(uint256)",
        0x625387DC: "unixTime()",
        0xE23CD19F: "writeJson(string,string)",
        0x35D6AD46: "writeJson(string,string,string)",
        0xC0865BA7: "writeToml(string,string)",
        0x51AC6A33: "writeToml(string,string,string)",
        0x14AE3519: "attachDelegation((uint8,bytes32,bytes32,uint64,address))",
        0xB25C5A25: "sign((address,uint256,uint256,uint256),bytes32)",
        0xC7FA7288: "signAndAttachDelegation(address,uint256)",
        0x5B593C7B: "signDelegation(address,uint256)",
        0x22100064: "rememberKey(uint256)",
        0xF0259E92: "breakpoint(string)",
        0xF7D39A8D: "breakpoint(string,bool)",
        0x203DAC0D: "copyStorage(address,address)",
        0x7404F1D2: "createWallet(string)",
        0x7A675BB6: "createWallet(uint256)",
        0xED7C5462: "createWallet(uint256,string)",
        0x6BCB2C1B: "deriveKey(string,string,uint32)",
        0x6229498B: "deriveKey(string,uint32)",
        0x28A249B0: "getLabel(address)",
        0xC6CE059D: "parseAddress(string)",
        0x974EF924: "parseBool(string)",
        0x8F5D232D: "parseBytes(string)",
        0x087E6E81: "parseBytes32(string)",
        0x42346C5E: "parseInt(string)",
        0xFA91454D: "parseUint(string)",
        0xDD82D13E: "skip(bool)",
        0x56CA623E: "toString(address)",
        0x71DCE7DA: "toString(bool)",
        0x71AAD10D: "toString(bytes)",
        0xB11A19E8: "toString(bytes32)",
        0xA322C40E: "toString(int256)",
        0x6900A3AE: "toString(uint256)",
        0x1206C8A8: "rpc(string,string)",
        0x975A6CE9: "rpcUrl(string)",
        0xA85A8418: "rpcUrls()",
        0x48C3241F: "closeFile(string)",
        0x261A323E: "exists(string)",
        0x7D15D019: "isDir(string)",
        0xE0EB04D4: "isFile(string)",
        0xC4BC59E0: "readDir(string)",
        0x60F9BB11: "readFile(string)",
        0x16ED7BC4: "readFileBinary(string)",
        0x70F55728: "readLine(string)",
        0x9F5684A2: "readLink(string)",
        0xF1AFE04D: "removeFile(string)",
        0x897E0A97: "writeFile(string,string)",
        0x619D897F: "writeLine(string,string)",
    }

    # bytes4(keccak256("assume(bool)"))
    assume_sig: int = 0x4C63E562

    # bytes4(keccak256("getCode(string)"))
    get_code_sig: int = 0x8D1CC925

    # bytes4(keccak256("prank(address)"))
    prank_sig: int = 0xCA669FA7

    # bytes4(keccak256("prank(address,address)"))
    prank_addr_addr_sig: int = 0x47E50CCE

    # bytes4(keccak256("startPrank(address)"))
    start_prank_sig: int = 0x06447D56

    # bytes4(keccak256("startPrank(address,address)"))
    start_prank_addr_addr_sig: int = 0x45B56078

    # bytes4(keccak256("stopPrank()"))
    stop_prank_sig: int = 0x90C5013B

    # bytes4(keccak256("deal(address,uint256)"))
    deal_sig: int = 0xC88A5E6D

    # bytes4(keccak256("store(address,bytes32,bytes32)"))
    store_sig: int = 0x70CA10BB

    # bytes4(keccak256("load(address,bytes32)"))
    load_sig: int = 0x667F9D70

    # bytes4(keccak256("fee(uint256)"))
    fee_sig: int = 0x39B37AB0

    # bytes4(keccak256("chainId(uint256)"))
    chainid_sig: int = 0x4049DDD2

    # bytes4(keccak256("coinbase(address)"))
    coinbase_sig: int = 0xFF483C54

    # bytes4(keccak256("difficulty(uint256)"))
    difficulty_sig: int = 0x46CC92D9

    # bytes4(keccak256("roll(uint256)"))
    roll_sig: int = 0x1F7B4F30

    # bytes4(keccak256("warp(uint256)"))
    warp_sig: int = 0xE5D6BF02

    # bytes4(keccak256("etch(address,bytes)"))
    etch_sig: int = 0xB4D6C782

    # bytes4(keccak256("ffi(string[])"))
    ffi_sig: int = 0x89160467

    # addr(uint256)
    addr_sig: int = 0xFFA18649

    # sign(uint256,bytes32)
    sign_sig: int = 0xE341EAA4

    # label(address,string)
    label_sig: int = 0xC657C718

    # bytes4(keccak256("getBlockNumber()"))
    get_block_number_sig: int = 0x42CBB15C

    # snapshotState()
    snapshot_state_sig: int = 0x9CD23835

    @staticmethod
    def handle(sevm, ex, arg: ByteVec, stack, step_id) -> ByteVec | None:
        funsig: int = int_of(arg[:4].unwrap(), "symbolic hevm cheatcode")
        ret = ByteVec()

        # vm.assert*
        if funsig in assert_cheatcode_handler:
            vm_assert = assert_cheatcode_handler[funsig](arg)
            not_cond = simplify(Not(vm_assert.cond))

            if ex.check(not_cond) != unsat:
                new_ex = sevm.create_branch(ex, not_cond, ex.pc)
                new_ex.halt(data=ByteVec(), error=FailCheatcode(f"{vm_assert}"))
                stack.push(new_ex, step_id)

            return ret

        # vm.assume(bool)
        elif funsig == hevm_cheat_code.assume_sig:
            assume_cond = simplify(is_non_zero(arg.get_word(4)))
            if is_false(assume_cond):
                raise InfeasiblePath("vm.assume(false)")
            ex.path.append(assume_cond, branching=True)
            return ret

        # vm.getCode(string)
        elif funsig == hevm_cheat_code.get_code_sig:
            path_len = arg.get_word(36)
            path = arg[68 : 68 + path_len].unwrap().decode("utf-8")

            if ":" in path:
                [filename, contract_name] = path.split(":")
                path = "out/" + filename + "/" + contract_name + ".json"

            target = sevm.options.root.rstrip("/")
            path = target + "/" + path

            with open(path) as f:
                artifact = json.loads(f.read())

            if artifact["bytecode"]["object"]:
                bytecode = artifact["bytecode"]["object"].replace("0x", "")
            else:
                bytecode = artifact["bytecode"].replace("0x", "")

            return stringified_bytes_to_bytes(bytecode)

        # vm.prank(address)
        elif funsig == hevm_cheat_code.prank_sig:
            sender = uint160(arg.get_word(4))
            result = ex.context.prank.prank(sender)
            if not result:
                raise HalmosException(
                    "can not call vm.prank(address) with an active prank"
                )
            return ret

        # vm.prank(address sender, address origin)
        elif funsig == hevm_cheat_code.prank_addr_addr_sig:
            sender = uint160(arg.get_word(4))
            origin = uint160(arg.get_word(36))
            result = ex.context.prank.prank(sender, origin)
            if not result:
                raise HalmosException(
                    "can not call vm.prank(address, address) with an active prank"
                )
            return ret

        # vm.startPrank(address)
        elif funsig == hevm_cheat_code.start_prank_sig:
            address = uint160(arg.get_word(4))
            result = ex.context.prank.startPrank(address)
            if not result:
                raise HalmosException(
                    "can not call vm.startPrank(address) with an active prank"
                )
            return ret

        # vm.startPrank(address sender, address origin)
        elif funsig == hevm_cheat_code.start_prank_addr_addr_sig:
            sender = uint160(arg.get_word(4))
            origin = uint160(arg.get_word(36))
            result = ex.context.prank.startPrank(sender, origin)
            if not result:
                raise HalmosException(
                    "can not call vm.startPrank(address, address) with an active prank"
                )
            return ret

        # vm.stopPrank()
        elif funsig == hevm_cheat_code.stop_prank_sig:
            ex.context.prank.stopPrank()
            return ret

        # vm.deal(address,uint256)
        elif funsig == hevm_cheat_code.deal_sig:
            who = uint160(arg.get_word(4))
            amount = uint256(arg.get_word(36))
            ex.balance_update(who, amount)
            return ret

        # vm.store(address,bytes32,bytes32)
        elif funsig == hevm_cheat_code.store_sig:
            if arg == hevm_cheat_code.fail_payload:
                # there isn't really a vm.fail() cheatcode, calling DSTest.fail()
                # really triggers vm.store(HEVM_ADDRESS, "failed", 1)
                # let's intercept it and raise an exception instead of actually storing
                # since HEVM_ADDRESS is an uninitialized account
                raise FailCheatcode()

            store_account = uint160(arg.get_word(4))
            store_slot = uint256(arg.get_word(36))
            store_value = uint256(arg.get_word(68))
            store_account_alias = sevm.resolve_address_alias(
                ex, store_account, stack, step_id, allow_branching=False
            )

            if store_account_alias is None:
                error_msg = f"vm.store() is not allowed for a nonexistent account: {hexify(store_account)}"
                raise HalmosException(error_msg)

            sevm.sstore(ex, store_account_alias, store_slot, store_value)
            return ret

        # vm.load(address,bytes32)
        elif funsig == hevm_cheat_code.load_sig:
            load_account = uint160(arg.get_word(4))
            load_slot = uint256(arg.get_word(36))
            load_account_alias = sevm.resolve_address_alias(
                ex, load_account, stack, step_id, allow_branching=False
            )

            if load_account_alias is None:
                # since load_account doesn't exist, its storage is empty.
                # note: the storage cannot be symbolic, as the symbolic storage cheatcode fails for nonexistent addresses.
                return ByteVec(con(0))

            return ByteVec(sevm.sload(ex, load_account_alias, load_slot))

        # vm.fee(uint256)
        elif funsig == hevm_cheat_code.fee_sig:
            ex.block.basefee = arg.get_word(4)
            return ret

        # vm.chainId(uint256)
        elif funsig == hevm_cheat_code.chainid_sig:
            ex.block.chainid = arg.get_word(4)
            return ret

        # vm.coinbase(address)
        elif funsig == hevm_cheat_code.coinbase_sig:
            ex.block.coinbase = uint160(arg.get_word(4))
            return ret

        # vm.difficulty(uint256)
        elif funsig == hevm_cheat_code.difficulty_sig:
            ex.block.difficulty = arg.get_word(4)
            return ret

        # vm.roll(uint256)
        elif funsig == hevm_cheat_code.roll_sig:
            ex.block.number = arg.get_word(4)
            return ret

        # vm.warp(uint256)
        elif funsig == hevm_cheat_code.warp_sig:
            ex.block.timestamp = arg.get_word(4)
            return ret

        # vm.etch(address,bytes)
        elif funsig == hevm_cheat_code.etch_sig:
            who = uint160(arg.get_word(4))

            # who must be concrete
            if not is_bv_value(who):
                error_msg = f"vm.etch(address who, bytes code) must have concrete argument `who` but received {who}"
                raise HalmosException(error_msg)

            # code must be concrete
            code_offset = int_of(arg.get_word(36), "symbolic code offset")
            code_length = int_of(arg.get_word(4 + code_offset), "symbolic code length")

            code_loc = 4 + code_offset + 32
            code_bytes = arg[code_loc : code_loc + code_length]
            ex.set_code(who, code_bytes)

            # vm.etch() initializes but does not clear storage
            ex.storage.setdefault(who, sevm.mk_storagedata())

            return ret

        # vm.ffi(string[]) returns (bytes)
        elif funsig == hevm_cheat_code.ffi_sig:
            if not sevm.options.ffi:
                error_msg = "ffi cheatcode is disabled. Run again with `--ffi` if you want to enable it"
                raise HalmosException(error_msg)

            cmd = extract_string_array_argument(arg, 0)

            if sevm.options.debug or sevm.options.verbose:
                print(f"[vm.ffi] {cmd}")

            process = Popen(cmd, stdout=PIPE, stderr=PIPE)
            (stdout, stderr) = process.communicate()

            if stderr:
                stderr_str = stderr.decode("utf-8")
                print(f"[vm.ffi] {cmd}, stderr: {red(stderr_str)}")

            out_str = stdout.decode("utf-8").strip()

            debug(f"[vm.ffi] {cmd}, stdout: {green(out_str)}")

            if decode_hex(out_str) is not None:
                # encode hex strings as is for compatibility with foundry's ffi
                pass
            else:
                # encode non-hex strings as hex
                out_str = out_str.encode("utf-8").hex()

            return stringified_bytes_to_bytes(out_str)

        # vm.addr(uint256 privateKey) returns (address keyAddr)
        elif funsig == hevm_cheat_code.addr_sig:
            private_key = uint256(extract_bytes(arg, 4, 32))

            # TODO: handle concrete private key (return directly the corresponding address)
            # TODO: check (or assume?) private_key is valid
            #  - less than curve order
            #  - not zero
            # TODO: add constraints that the generated addresses are reasonable
            #  - not zero
            #  - not the address of a known contract

            addr = apply_vmaddr(ex, private_key)
            ret.append(uint256(addr))
            return ret

        # vm.sign(uint256 privateKey, bytes32 digest) returns (uint8 v, bytes32 r, bytes32 s)
        elif funsig == hevm_cheat_code.sign_sig:
            key = extract_bytes(arg, 4, 32)
            digest = extract_bytes(arg, 4 + 32, 32)

            # TODO: handle concrete private key + digest (generate concrete signature)

            # check for an existing signature
            known_sigs = ex.known_sigs
            (v, r, s) = known_sigs.get((key, digest), (None, None, None))
            if (v, r, s) == (None, None, None):
                # if not, create a new signature
                v, r, s = (f(key, digest) for f in (f_sign_v, f_sign_r, f_sign_s))

                # associate the new signature with the private key and digest
                known_sigs[(key, digest)] = (v, r, s)

                # constrain values to their expected ranges
                in_range = And(
                    Or(v == 27, v == 28),
                    ULT(0, r),
                    ULT(r, secp256k1n),
                    ULT(0, s),
                    ULT(s, secp256k1n),
                )
                ex.path.append(in_range)

                # explicitly model malleability
                recover = f_ecrecover(digest, v, r, s)
                recover_malleable = f_ecrecover(digest, v ^ 1, r, secp256k1n - s)

                addr = apply_vmaddr(ex, key)
                ex.path.append(recover == addr)
                ex.path.append(recover_malleable == addr)

                # mark signatures as distinct if key or digest are distinct
                # NOTE: the condition `And(r != _r, s != _s)` is stronger than `Or(v != _v, r != _r, s != _s)` which is sound
                # TODO: we need to figure out whether this stronger condition is necessary and whether it could lead to unsound results in practical cases
                for (_key, _digest), (_v, _r, _s) in known_sigs.items():
                    distinct = Implies(
                        Or(key != _key, digest != _digest),
                        Or(v != _v, r != _r, s != _s),
                    )
                    ex.path.append(distinct)

            ret.append(uint256(v))
            ret.append(r)
            ret.append(s)
            return ret

        # vm.label(address account, string calldata newLabel)
        elif funsig == hevm_cheat_code.label_sig:
            addr = extract_bytes(arg, 4, 32)

            # TODO: no-op for now
            # label = extract_string_argument(arg, 1)

            return ret

        # vm.getBlockNumber() return (uint256)
        elif funsig == hevm_cheat_code.get_block_number_sig:
            ret.append(uint256(ex.block.number))
            return ret

        # vm.snapshotState() return (uint256)
        elif funsig == hevm_cheat_code.snapshot_state_sig:
            return snapshot_state(ex, arg, sevm, stack, step_id)

        elif funsig in hevm_cheat_code.dict_of_unsupported_cheatcodes:
            msg = f"Unsupported cheat code: {hevm_cheat_code.dict_of_unsupported_cheatcodes[funsig]}"
            raise HalmosException(msg)

        else:
            # TODO: support other cheat codes
            msg = f"Unsupported cheat code: calldata = {hexify(arg)}"
            raise HalmosException(msg)
