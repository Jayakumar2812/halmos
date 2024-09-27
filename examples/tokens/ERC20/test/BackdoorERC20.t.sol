// SPDX-License-Identifier: AGPL-3.0
pragma solidity >=0.8.0 <0.9.0;

import {ERC20Test} from "./ERC20Test.sol";

import {BackdoorERC20} from "../src/BackdoorERC20.sol";

import {IERC20} from "forge-std/interfaces/IERC20.sol";

// empty interface
interface IEmpty { }

/// @custom:halmos --solver-timeout-assertion 0
contract BackdoorERC20Test is ERC20Test {

    /// @custom:halmos --solver-timeout-branching 1000
    function setUp() public override {
        address deployer = address(0x1000);

        BackdoorERC20 token_ = new BackdoorERC20("BackdoorERC20", "BackdoorERC20", 1_000_000_000e18, deployer);
        token = address(token_);

        holders = new address[](3);
        holders[0] = address(0x1001);
        holders[1] = address(0x1002);
        holders[2] = address(0x1003);

        for (uint i = 0; i < holders.length; i++) {
            address account = holders[i];
            uint256 balance = svm.createUint256('balance');
            vm.prank(deployer);
            token_.transfer(account, balance);
            for (uint j = 0; j < i; j++) {
                address other = holders[j];
                uint256 amount = svm.createUint256('amount');
                vm.prank(account);
                token_.approve(other, amount);
            }
        }
    }

    function check_NoBackdoor_with_createBytes(bytes4 selector, address caller, address other) public {
        // arbitrary bytes as calldata
        bytes memory args = svm.createBytes(1024, 'data');
        bytes memory data = abi.encodePacked(selector, args);
        _checkNoBackdoor(data, caller, other); // backdoor counterexample
    }

    function check_NoBackdoor_with_createCalldata_BackdoorERC20(bytes4 selector, address caller, address other) public {
        // calldata created using explicit BackdoorERC20 abi
        bytes memory data = svm.createCalldata("BackdoorERC20");
        vm.assume(selector == bytes4(data)); // to enhance counterexample
        _checkNoBackdoor(data, caller, other); // backdoor counterexample
    }

    // NOTE: a backdoor counterexample can be found even if the abi information used to generate calldata doesn't include the backdoor function.
    // This is because the createCalldata cheatcode also generates fallback calldata which can match any other functions in the target contract.
    //
    // Caveat: the fallback calldata is essentially the same as the arbitrary bytes generated by the createBytes() cheatcode.
    // This means that if the functions matched by fallback calldata have dynamic-sized parameters, symbolic calldata offset errors may occur.
    // The main advantage of using createCalldata() is its more reliable handling of dynamic-sized parameters, which helps to avoid such symbolic offset errors.

    function check_NoBackdoor_with_createCalldata_IERC20(bytes4 selector, address caller, address other) public {
        // calldata created using the standard ERC20 interface abi
        bytes memory data = svm.createCalldata("IERC20");
        vm.assume(selector == bytes4(data)); // to enhance counterexample
        _checkNoBackdoor(data, caller, other); // backdoor counterexample
    }

    function check_NoBackdoor_with_createCalldata_IEmpty(bytes4 selector, address caller, address other) public {
        // calldata created using an empty interface
        bytes memory data = svm.createCalldata("IEmpty");
        vm.assume(selector == bytes4(data)); // to enhance counterexample
        _checkNoBackdoor(data, caller, other); // backdoor counterexample
    }
}
