"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      EPOS_drv.py
brief:     This module contains classes and methods to work with EPOS controllers.
author(s): Jost Prevc, Janez Bozja
date:      10.11.2017

details:   This module contains classes and methods to work with EPOS2/4 controllers.
"""

import ctypes
import os
import time
import platform
import typing
from typing import List, Tuple, Union, Dict, Optional


class EPOSMotor:
    """
    EPOS_Motor object initialization function.

    :param open_dlg: If True, device dialog window will be opened for choosing the device for connection.
    :param node_id: Motor's node number to be assigned to device.
    :param epos_version: Version of EPOS controller to connect to. Currently supported are 'EPOS2' and 'EPOS4'.
                         If not provided, connection will be established to the first found EPOS device.
    :param motor_name: User defined name for the motor.
    :param max_pos_mode_velocity: Maximum velocity (rpm) in profile position mode. If None,
                                  this parameter will be read from controller.
    :param protocol_stack_name: Name of used communication protocol. Can be one of the following: 'MAXON_RS232',
                                'MAXON SERIAL V2' or 'CANopen'.m acceleration: Acceleration value in profile.
                                If None, this parameter will be read from controller.
    :param acceleration: TODO Add attribute description.
    :param port_name: Name of USB port to be connected to. If left None, port names will automatically be acquired and
                      connection will be made to the first one found.
    :param connect_on_init: If True, connection will be established upon creation of the object. If False, object will
                            only be created but connection to the device won't be established yet. In that case it
                            should be established explicitly by calling :meth:`connect` method.
    """

    def __init__(
        self,
        open_dlg: bool = False,
        node_id: int = 0,
        epos_version: Optional[str] = None,
        motor_name: str = "EPOS Maxon Motor",
        max_pos_mode_velocity: int = 100,
        acceleration: int = 300,
        protocol_stack_name: str = "MAXON SERIAL V2",
        port_name: Optional[str] = None,
        connect_on_init: bool = True,
    ):
        self.open_dlg = open_dlg
        self.epos_version = epos_version
        self.max_pos_mode_velocity = max_pos_mode_velocity
        self.acceleration = acceleration
        self.protocol_stack_name = protocol_stack_name
        self.port_name = port_name

        # get path to folder containing this module
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # Add path to DLL to path environmental variable and load DLL
        architecture, _ = platform.architecture()

        dll_folder_abs_path = os.path.join(dir_path, "EPOS_DLL")

        # add folder which contains unicorn.dll and dependent dlls to path
        if dll_folder_abs_path not in os.environ["PATH"]:
            os.environ["PATH"] = dll_folder_abs_path + os.pathsep + os.environ["PATH"]

        # open dll
        if architecture == "64bit":
            self.windll = ctypes.WinDLL(dll_folder_abs_path + "\EposCmd64.dll")
        elif architecture == "32bit":
            self.windll = ctypes.WinDLL("EposCmd.dll")

        self.node_id = node_id
        self.motor_name = motor_name

        self.error_code = ctypes.c_uint32(0)  # stores last recorded error code

        # set argument and return types for functions in DLL Epos library
        self._set_dll_argres_types()

        self._dev_handle = ctypes.c_void_p()

        if connect_on_init:
            self.connect()

    def __repr__(self) -> str:
        """
        Representation object. Provides explanation on motor object when it
        gets called.
        """

        motor_status = self.get_state()
        op_mode = self._op_mode_table[self.get_operation_mode()]

        return (
            self.motor_name
            + "\n"
            + "Node:           "
            + str(self.node_id)
            + "\n"
            + "Motor state:    "
            + str(motor_status)
            + "\n"
            + "Operation mode: "
            + op_mode
            + "\n"
            + "Position:       "
            + str(self.get_motor_position())
            + " steps\n"
            + "Velocity:       "
            + str(self.get_averaged_velocity())
            + " rpm\n"
            + "Max velocity (in position mode): "
            + str(self.get_max_velocity())
            + " rpm\n"
        )

    def connect(self) -> None:
        """
        Connects to the device.
        """
        # get device handle and establish connection
        self._dev_handle = self._get_device_handle(self.open_dlg)

        # clear fault state (if it is set)
        self.clear_fault()

        # set enable state
        self.set_enable_state()

        # set profile position mode
        self.activate_profile_position_mode()

        # configure max velocity, acceleration and deceleration (initially, acceleration and deceleration
        # have the same value)
        self.set_position_profile(self.max_pos_mode_velocity, self.acceleration, self.acceleration)

    def open_device_dialog(self) -> ctypes.c_void_p:
        """
        Opens device dialog window which prompts user to select communication
        parameters (Device Name, Protocol stack name, Interface name,
        Port name, Baudrate, Timeout)

        :return: Device handle.

        :raises EposException: If there was problem while communicating with controller.
        """
        ret_handle = self.windll.VCS_OpenDeviceDlg(ctypes.byref(self.error_code))

        if not ret_handle:
            raise EPOSMotor.EposException(
                f"EPOS: Error while opening device.\n"
                f"EPOS Error code: 0x{self.error_code.value:08x}\n"
                f"EPOS Error message: {self._get_error_info(self.error_code)}\n"
                f"Check connection with PC."
            )

        return ret_handle

    def open_device(
        self,
        dev_name: str = "EPOS2",
        protocol_stack_name: str = "MAXON SERIAL V2",
        interface_name: str = "USB",
        port_name: str = "USB0",
    ) -> ctypes.c_void_p:
        """
        Opens device and returns its handle.

        :param dev_name: Device's name
        :param protocol_stack_name: Protocol stack name
        :param interface_name: Interface name
        :param port_name: Port name

        :return: handle object

        :raises EposException: If there was problem while communicating with controller.
        """
        ret_handle = self.windll.VCS_OpenDevice(
            dev_name.encode("utf-8"),
            protocol_stack_name.encode("utf-8"),
            interface_name.encode("utf-8"),
            port_name.encode("utf-8"),
            ctypes.byref(self.error_code),
        )

        if not ret_handle:
            raise EPOSMotor.EposException(
                f"EPOS: Error while opening device.\n"
                f"EPOS Error code: 0x{self.error_code.value:08x}\n"
                f"EPOS Error message: {self._get_error_info(self.error_code)}\n"
                f"Check connection with PC."
            )

        return ret_handle

    def close_device(self) -> None:
        """
        Closes device.

        :raises EposException: If there was problem while communicating with controller.
        """

        retval = self.windll.VCS_CloseDevice(self._dev_handle, ctypes.byref(self.error_code))

        self._check_retval(retval)

    def get_port_name_selection(self, dev_name: str, protocol_stack_name: str, interface_name: str) -> List[str]:
        """
        Returns names of communication ports in a list.

        :param dev_name: Name of the device.
        :param protocol_stack_name: Name of used communication protocol. Can be one of the following: 'MAXON_RS232',
                                    'MAXON SERIAL V2' or 'CANopen'.
        :param interface_name: Name of interface.

        :return: List of USB ports.

        :raises EPOSMotor.EposException: If port name selection could not be retrieved.
        """
        port_selection = (ctypes.c_char * 200)()
        end_of_selection = ctypes.c_bool(False)

        port_selection_lst = []
        get_first_selection = True
        while not end_of_selection:
            retval = self.windll.VCS_GetPortNameSelection(
                dev_name.encode("utf-8"),
                protocol_stack_name.encode("utf-8"),
                interface_name.encode("utf-8"),
                ctypes.c_bool(get_first_selection),
                port_selection,
                ctypes.c_uint16(200),
                ctypes.byref(end_of_selection),
                ctypes.byref(self.error_code),
            )

            self._check_retval(retval)

            get_first_selection = False

            port_selection_lst.append(port_selection.value.decode("utf8"))

        return port_selection_lst

    def set_enable_state(self) -> None:
        """
        Sets enable state for the motor. Enable state is necessary for motor to
        be allowed to move.

        :raises EposException: If there was problem while communicating with controller.
        """
        retval = self.windll.VCS_SetEnableState(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

    def set_disable_state(self) -> None:
        """
        Sets disable state for the motor. Disable state prevents the motor
        from movement. In disable state, motor will not acknowledge any
        movement commands.

        :raises EposException: If there was problem while communicating with controller.
        """

        retval = self.windll.VCS_SetDisableState(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

    def move_to_position(self, position: int, absolute: bool = True, immediately: bool = True) -> None:
        """
        Moves motor to new position.

        :param position: New motor position in steps (4000 encoder steps = 1 rotation)
        :param absolute: Specifies whether the movement will be done as absolute or relative
                         position.
        :param immediately: Specifies whether the movement will start immediately. If True,
                            motor will start moving towards new target even if the previous
                            target was not yet reached. If False, motor will wait until current
                            target is reached and then move to specified position.

        :raises PermissionError: If EPOS is currently not in position mode.
        :raises EposException: If there was problem while communicating with controller.

        """
        if self.get_operation_mode() != 1:
            raise PermissionError(
                "EPOS: Motor operating state not in profile "
                + "position mode. Refer to "
                + "activate_profile_position_mode."
            )

        retval = self.windll.VCS_MoveToPosition(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.c_long(position),
            ctypes.c_bool(absolute),
            ctypes.c_bool(immediately),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

    def reset_position(self) -> None:
        """
        Resets position counter to zero.
        """
        self.set_position_count(0)

    def set_position_count(self, pos_count: int) -> None:
        """
        Sets new position value for motor encoder. When setting position, target velocity or position
        should first be reached.

        :param pos_count: New position value

        :raises EposException: If there was problem while communicating with controller or if target
                               velocity or position was not yet reached when setting position.
        """

        if not self.is_target_reached:
            raise EPOSMotor.EposException(
                "EPOS: Error while setting position count.\n"
                + "Setting motor position should be done only when "
                + "Target velocity or position is reached "
                + "(is_target_reached == True)"
            )

        retval = self.windll.VCS_DefinePosition(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.c_long(pos_count), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

    def write_device_parameters(self, file_path: str) -> None:
        r"""
        Writes parameters from a file to the device.

        :param file_path: File path to \*.dcf file which contains device parameters

        :raises EposException:
        """
        self.set_disable_state()
        retval = self.windll.VCS_ImportParameter(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            file_path.encode("utf-8"),
            ctypes.c_bool(False),
            ctypes.c_bool(False),
            ctypes.byref(self.error_code),
        )
        self._check_retval(retval)
        self.set_enable_state()

    def read_device_parameters(self, file_path: str) -> None:
        r"""
        Reads all device parameters and writes them to the file.

        :param file_path: File path to \*.dcf file which contains device parameters

        :raises EposException:
        """
        version = self.get_version()
        fw_version = hex(version[1]) + "_" + hex(version[0])
        firmware_file_names = {"0x2126_0x6220": "Epos_2126h_6220h_0000h_0000h.bin"}
        firmware_file_name = firmware_file_names[fw_version]
        user_id = ""
        comment = ""

        retval = self.windll.VCS_ExportParameter(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            file_path.encode("utf-8"),
            firmware_file_name.encode("utf-8"),
            user_id.encode("utf-8"),
            comment.encode("utf-8"),
            ctypes.c_bool(False),
            ctypes.c_bool(False),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

    def get_motor_parameters(self) -> List[int]:
        """
        Receives motor parameters (Nominal current, Maximum output current,
        Thermal time constant)

        :return: List of parameters : [nominalCurrent, maxOutputCurrent, thermalTimeConst]

        :raises EposException: If there was problem while communicating with controller.
        """

        nom_current = ctypes.c_uint16(0)
        max_output_curr = ctypes.c_uint16(0)
        therm_time_const = ctypes.c_uint16(0)

        retval = self.windll.VCS_GetEcMotorParameter(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.byref(nom_current),
            ctypes.byref(max_output_curr),
            ctypes.byref(therm_time_const),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

        return [nom_current.value, max_output_curr.value, therm_time_const.value]

    def get_encoder_parameters(self) -> Tuple[int, bool]:
        """
        Returns encoder parameters, which are currently set on controller (pulse per rotation, inverted polarity).

        ..note:: Pulses per rotation value is number of pulses of A/B incremental lines per one mechanical revolution.
                 For quadrature value, simply multiply returned value with 4.

        :return: Tuple of two values. First value is number of incremental pulses per one mechanical revolution. Second
                 value is a boolean value, which is set to True if encoder polarity is inverted and False otherwise.
        """

        pulse_number = ctypes.c_uint32(0)
        inverted_polarity = ctypes.c_bool(False)

        retval = self.windll.VCS_GetIncEncoderParameter(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.byref(pulse_number),
            ctypes.byref(inverted_polarity),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

        return pulse_number.value, inverted_polarity.value

    def get_motor_position(self) -> int:
        """
        Reads position from motor encoder.

        :return: Motor position value in encoder steps.

        :raises EposException: If there was problem while communicating with controller.
        """
        position = ctypes.c_long(0)

        retval = self.windll.VCS_GetPositionIs(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(position), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

        return position.value

    def get_target_motor_position(self) -> int:
        """
        Reads target motor position which was commanded to controller.

        :return: Target motor position value in encoder steps.

        :raises EposException: If there was problem while communicating with controller.
        """
        target_position = ctypes.c_long(0)

        retval = self.windll.VCS_GetTargetPosition(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.byref(target_position),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

        return target_position.value

    def get_state(self, str_repr: bool = True) -> Union[str, int]:
        """
        Reads motor state.

        :param str_repr: Specifies whether output state should be provided as string.

        :return: Received motor state.

        **State code**

        | 0 - DISABLED
        | 1 - ENABLED
        | 2 - QUICKSTOP
        | 3 - FAULT

        :raises EposException: If there was problem while communicating with controller.

        """

        state_int = ctypes.c_uint16(0)
        retval = self.windll.VCS_GetState(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(state_int), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

        if str_repr:
            state_to_str_map = {0: "DISABLED", 1: "ENABLED", 2: "QUICKSTOP", 3: "FAULT"}
            return state_to_str_map.get(state_int.value, "UNKNOWN STATE")
        return state_int.value

    def wait_for_target_reached(self, timeout: int) -> None:
        """
        Waits until target position (in position operating mode) or
        velocity (in velocity operation mode) is reached, blocks the program flow.

        :param timeout: Maximum time to wait in milliseconds.

        :raises ValueError: If 'timeout' parameter is not greater than 0.
        :raises TimeoutError: If motor was unable to reach target in specified time.
        :raises EposException: If there was problem while communicating with controller.
        """

        if timeout <= 0:
            raise ValueError("Parameter 'timeout' should be a positive integer.")

        retval = self.windll.VCS_WaitForTargetReached(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.c_uint32(timeout), ctypes.byref(self.error_code)
        )

        if self.error_code.value == 0x1000000B:
            raise TimeoutError("EPOS: Target not reached in specified timeout.")

        self._check_retval(retval)

    def clear_fault(self) -> None:
        """
        Changes the device state from 'FAULT' to 'DISABLE'.

        :raises EposException: If there was problem while communicating with controller.
        """

        retval = self.windll.VCS_ClearFault(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

    def get_operation_mode(self) -> int:
        """
        Receives operation mode information.

        :return: Operation mode code (refer to EPOS command library, operation mode, sec. 5, page 47).

        :raises EposException: If there was problem while communicating with controller.

        """

        op_mode = ctypes.c_int8(0)

        retval = self.windll.VCS_GetOperationMode(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(op_mode), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

        return op_mode.value

    def _set_operation_mode(self, op_mode: int) -> None:
        """
        Sets operation mode information.

        :param op_mode: Operation mode code (refer to EPOS command library - operation mode)

        :raises EposException: If there was problem while communicating with controller.
        """

        retval = self.windll.VCS_SetOperationMode(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.c_int8(op_mode), ctypes.byref(self.error_code)
        )

        time.sleep(0.2)

        self._check_retval(retval)

    def get_movement_state(self) -> bool:
        """
        Receives movement state from device.

        :return: True, if target position/velocity reached, False otherwise.

        :raises EposException: If there was problem while communicating with controller.
        """
        move_state = ctypes.c_bool(True)

        retval = self.windll.VCS_GetMovementState(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(move_state), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

        return move_state.value

    def move_with_velocity(self, velocity: int) -> None:
        """
        Moves with specified velocity.

        :param velocity: Specified velocity in rpm

        :raises PermissionError: If motor is not in profile velocity mode when issuing the command.
        :raises EposException: If there was problem while communicating with controller.
        """

        if self.get_operation_mode() != 3:
            raise PermissionError(
                "EPOS: Motor operating state not in profile "
                + "velocity mode. Refer to "
                + "activate_profile_velocity_mode."
            )

        retval = self.windll.VCS_MoveWithVelocity(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.c_long(velocity),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

    def soft_stop(self) -> None:
        """
        Stops motor movement with deceleration.

        :raises EposException: If there was problem while communicating with controller.
        """

        retval = self.windll.VCS_HaltVelocityMovement(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

    def get_velocity(self) -> int:
        """
        Reads current velocity and returns value in rpm.

        :return: Velocity value in rpm.

        :raises EposException: If there was problem while communicating with controller.
        """
        velocity = ctypes.c_long(0)

        retval = self.windll.VCS_GetVelocityIs(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(velocity), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

        return velocity.value

    def get_averaged_velocity(self) -> int:
        """
        Reads averaged velocity and returns value in rpm.

        :return: Averaged velocity value in rpm.

        :raises EposException: If there was problem while communicating with controller.
        """

        velocity = ctypes.c_long(0)

        retval = self.windll.VCS_GetVelocityIsAveraged(
            self._dev_handle, ctypes.c_uint16(self.node_id), ctypes.byref(velocity), ctypes.byref(self.error_code)
        )

        self._check_retval(retval)

        return velocity.value

    def get_target_velocity(self) -> int:
        """
        Returns target velocity, that is velocity, which was last commanded to controller.

        :return: Target velocity in rpm.
        """

        target_velocity = ctypes.c_long(0)

        retval = self.windll.VCS_GetTargetVelocity(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.byref(target_velocity),
            ctypes.byref(self.error_code),
        )
        self._check_retval(retval)

        return target_velocity.value

    def activate_profile_position_mode(self) -> None:
        """
        Activates profile positioning operation mode. In profile positioning mode, controller's regulation loop will
        work to regulate motor's position on the target value. That means that all position-based methods should be
        done when this mode is active.
        """

        self._set_operation_mode(1)

    def activate_profile_velocity_mode(self) -> None:
        """
        Activates profile velocity operation mode. In profile velocity mode, controller's regulation loop will
        work to regulate motor's velocity on the target value. That means that all velocity-based methods should be
        done when this mode is active.
        """

        self._set_operation_mode(3)

    def activate_profile_current_mode(self) -> None:
        """
        Activates profile current operation mode.
        """

        self._set_operation_mode(-3)

    def get_current_must(self) -> int:
        """
        Returns the current mode setting value.

        :return: Current mode setting value.
        """
        current_must = ctypes.c_short(0)

        retval = self.windll.VCS_GetCurrentMust(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.byref(current_must),
            ctypes.byref(self.error_code),
        )
        self._check_retval(retval)

        return current_must.value

    def set_current_must(self, current: int) -> None:
        """
        Sets the current mode setting value.

        :param current: Current mode setting value
        """
        retval = self.windll.VCS_SetCurrentMust(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.c_short(current),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

    def get_position_profile(self) -> List[int]:
        """
        Receives position profile parameters.

        List of position profile parameters
        [Velocity, Acceleration, Deceleration]

        :return: Position profile parameters: [velocity, acceleration, deceleration]

        :raises EposException: If there was problem while communicating with controller.
        """

        profile_velocity = ctypes.c_uint32(0)
        profile_acceleration = ctypes.c_uint32(0)
        profile_deceleration = ctypes.c_uint32(0)

        retval = self.windll.VCS_GetPositionProfile(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.byref(profile_velocity),
            ctypes.byref(profile_acceleration),
            ctypes.byref(profile_deceleration),
            ctypes.byref(self.error_code),
        )
        self._check_retval(retval)

        return [profile_velocity.value, profile_acceleration.value, profile_deceleration.value]

    def set_position_profile(
        self,
        profile_velocity: Optional[int] = None,
        profile_acceleration: Optional[int] = None,
        profile_deceleration: Optional[int] = None,
    ) -> None:
        """int
        Sets position profile parameters for trapezoidal profile.
        Note: If any of parameters is not provided, the parameter value will remain the same.

        :param profile_velocity: Profile maximum velocity to be reached
        :param profile_acceleration: Profile acceleration
        :param profile_deceleration: Profile deceleration

        :raises ValueError: If any of profile parameters is not greater than zero.
        :raises EposException: If there was problem while communicating with controller.
        """

        profile_to_set = [profile_velocity, profile_acceleration, profile_deceleration]

        for profile_val in profile_to_set:
            if profile_val is not None and profile_val <= 0:
                raise ValueError("Specified profile value should be a positive integer.")

        is_set = [param is not None for param in profile_to_set]

        if not all(is_set):
            # read current values
            read_profile = self.get_position_profile()

            for i in range(3):
                # if value was not specified, set it as it was
                profile_to_set[i] = profile_to_set[i] if is_set[i] else read_profile[i]

        max_velocity, acc, dec = (typing.cast(int, profile_to_set[i]) for i in range(3))

        retval = self.windll.VCS_SetPositionProfile(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.c_uint32(max_velocity),
            ctypes.c_uint32(acc),
            ctypes.c_uint32(dec),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

    def set_max_velocity(self, max_velocity: int) -> None:
        """
        Sets maximum motor velocity in positional movement profile.
        Acceleration and deceleration parameters remain the same. This will
        not affect behaviour in Profile velocity mode.

        :param max_velocity: Maximum motor velocity when moving in rpm.
        """

        profile = self.get_position_profile()
        self.set_position_profile(max_velocity, profile[1], profile[2])

    def get_max_velocity(self) -> int:
        """
        Returns maximum velocity in position profile mode.

        :return: Maximum velocity value in position profile mode in rpm.
        """

        profile = self.get_position_profile()

        return profile[0]

    def wait_for_speed_reached(self, timeout: float, err_thr: float = 5, throw: bool = True) -> None:
        """
        Wait for motor to reach its target velocity.

        Note: This function is needed because it was observed that motor.wait_for_target_reached
        function returned sooner than expected. Use this function for a more robust operation.

        :param err_thr: Error threshold in rpm, which is allowed velocity error to break from this function.
        :param timeout: Maximum time to wait in this function in milliseconds. After this time is exceeded,
                        function will return, or raise exception, if 'throw' is set to True.
        :param throw: Specifies, whether RuntimeError should be raised if timeout is exceeded.

        :raise RuntimeError: If motor does not stop in specified time.

        NOTE: due to the issues with heavy loads, under/over-flow velocity weren't detected by previous design.
            New design solves this by repeating following sequence, after short-averaged velocity
            (`self.get_averaged_velocity()`) is first within err_thr:
            - read averaged velocities multiple times for step_time_sec seconds
            - calculate diff (`abs(maximum - minimum)`) of averaged velocity values.
            - check if dif < err_thr*2.
        """
        step_time_sec = 0.5
        target_velocity = self.get_target_velocity()

        end_timestamp = time.time() + (timeout / 1000)
        while time.time() < end_timestamp:
            if abs(self.get_averaged_velocity() - target_velocity) <= err_thr:
                # stage 2, see note above
                while time.time() < end_timestamp:
                    sec_end_timestamp = time.time() + step_time_sec
                    velocities = []
                    while time.time() < sec_end_timestamp:
                        velocities.append(self.get_averaged_velocity())

                    diff = abs(max(velocities) - min(velocities))
                    if diff < (err_thr * 2):
                        return  # success, velocity is within specified range

        if throw:
            raise RuntimeError(
                f"Motor did not reach target velocity {target_velocity} in specified time ({timeout} ms)."
            )

    def get_version(self) -> Tuple[int, int, int, int]:
        """
        Returns controller's firmware version.

        :return: Tuple of four values - hardware version, software version, application number
                 and application version
        """

        hardware_version = ctypes.c_uint16()
        software_version = ctypes.c_uint16()
        app_number = ctypes.c_uint16()
        app_version = ctypes.c_uint16()

        retval = self.windll.VCS_GetVersion(
            self._dev_handle,
            ctypes.c_uint16(self.node_id),
            ctypes.byref(hardware_version),
            ctypes.byref(software_version),
            ctypes.byref(app_number),
            ctypes.byref(app_version),
            ctypes.byref(self.error_code),
        )

        self._check_retval(retval)

        return hardware_version.value, software_version.value, app_number.value, app_version.value

    def get_device_name_selection(self) -> List[str]:
        """
        Returns a list of possible EPOS devices.

        :return: List of all possible EPOS devices, that can be passed to connection functions as device names.
        """

        max_device_name_length = 100
        device_name = (ctypes.c_char * max_device_name_length)()
        end_of_selection = ctypes.c_bool(False)

        device_name_selection = []
        get_first_selection = True
        while not end_of_selection:
            retval = self.windll.VCS_GetDeviceNameSelection(
                ctypes.c_bool(get_first_selection),
                device_name,
                ctypes.c_uint16(max_device_name_length),
                ctypes.byref(end_of_selection),
                ctypes.byref(self.error_code),
            )

            self._check_retval(retval)

            get_first_selection = False

            device_name_selection.append(device_name.value.decode("utf8"))

        return device_name_selection

    def _get_possible_device_names(self) -> List[str]:
        if self.epos_version is None:
            device_names = self.get_device_name_selection()
        else:
            device_names = [self.epos_version]

        return device_names

    def _get_device_name_to_port_list_map(self) -> Dict[str, List[str]]:
        name_to_port_map = {}

        for device_name in self.get_device_name_selection():
            if self.port_name is None:
                try:
                    port_names = self.get_port_name_selection(device_name, self.protocol_stack_name, "USB")
                except EPOSMotor.EposException:
                    port_names = []
            else:
                port_names = [self.port_name]

            name_to_port_map[device_name] = port_names

        return name_to_port_map

    def _get_device_handle(self, is_open_dlg: bool) -> ctypes.c_void_p:
        if is_open_dlg:
            return self.open_device_dialog()

        dev_names_to_port_map = self._get_device_name_to_port_list_map()

        for dev_name, port_list in dev_names_to_port_map.items():
            for port in port_list:
                try:
                    dev_handle = self.open_device(dev_name=dev_name, port_name=port)
                except EPOSMotor.EposException:
                    pass
                else:
                    return dev_handle

        raise EPOSMotor.EposException(
            "Could not connect to EPOS device. Check USB connection and power supply and make sure that no"
            "other application is connected to EPOS."
        )

    def _get_error_info(self, err_code: ctypes.c_uint32) -> str:

        max_err_info_length = 200
        err_info = (ctypes.c_char * max_err_info_length)()

        self.windll.VCS_GetErrorInfo(err_code, err_info, max_err_info_length)

        return err_info.value.decode("utf8")

    @property
    def is_target_reached(self) -> bool:
        """
        Returns True, if motor already reached its commanded target, and False
        otherwise.

        :return: True, if motor already reached its commanded target, and False
                 otherwise.
        """
        return self.get_movement_state()

    _op_mode_table = {
        1: "Profile Position Mode",
        3: "Profile Velocity Mode",
        6: "Homing Mode",
        7: "Interpolated Position Mode",
        -1: "Position Mode",
        -2: "Velocity Mode",
        -3: "Current Mode",
        -5: "Master Encoder Mode",
        -6: "Step Direction Mode",
    }

    class EposException(Exception):
        """
        EposException implementation of Exception.
        """

    def _check_retval(self, retval: int) -> None:
        if not retval:
            raise EPOSMotor.EposException(
                f"EPOS Error code: 0x{self.error_code.value:08x}\n"
                f"EPOS Error message: {self._get_error_info(self.error_code)}"
            )

    def _set_dll_argres_types(self) -> None:
        """
        Function parameters definitions from EPOS DLL. Defines input and output types.
        """
        # pylint: disable=too-many-statements

        self.windll.VCS_OpenDeviceDlg.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.windll.VCS_OpenDeviceDlg.restype = ctypes.c_void_p

        self.windll.VCS_OpenDevice.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_OpenDevice.restype = ctypes.c_void_p

        self.windll.VCS_CloseDevice.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        self.windll.VCS_CloseDevice.restype = ctypes.c_bool

        self.windll.VCS_GetVersion.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetVersion.restype = ctypes.c_bool

        self.windll.VCS_GetPortNameSelection.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.c_char_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetPortNameSelection.restype = ctypes.c_bool

        self.windll.VCS_GetEcMotorParameter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetEcMotorParameter.restype = ctypes.c_bool

        self.windll.VCS_GetIncEncoderParameter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetIncEncoderParameter.restype = ctypes.c_bool

        self.windll.VCS_GetPositionIs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetPositionIs.restype = ctypes.c_bool

        self.windll.VCS_GetTargetPosition.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetTargetPosition.restype = ctypes.c_bool

        self.windll.VCS_MoveToPosition.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.c_long,  # target position
            ctypes.c_bool,  # absolute
            ctypes.c_bool,  # immediately
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error code
        self.windll.VCS_MoveToPosition.restype = ctypes.c_bool

        self.windll.VCS_GetState.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.POINTER(ctypes.c_uint16),  # state
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error
        self.windll.VCS_GetState.restype = ctypes.c_bool

        self.windll.VCS_SetEnableState.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error
        self.windll.VCS_SetEnableState.restype = ctypes.c_bool

        self.windll.VCS_SetDisableState.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error
        self.windll.VCS_SetDisableState.restype = ctypes.c_bool

        self.windll.VCS_WaitForTargetReached.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.c_uint32,  # timeout
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error
        self.windll.VCS_WaitForTargetReached.restype = ctypes.c_bool

        self.windll.VCS_ClearFault.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error
        self.windll.VCS_ClearFault.restype = ctypes.c_bool

        self.windll.VCS_SetQuickStopState.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error
        self.windll.VCS_SetQuickStopState.restype = ctypes.c_bool

        self.windll.VCS_GetOperationMode.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_uint16,  # node id
            ctypes.POINTER(ctypes.c_int8),  # mode
            ctypes.POINTER(ctypes.c_uint32),
        ]  # error
        self.windll.VCS_GetOperationMode.restype = ctypes.c_bool

        self.windll.VCS_GetMovementState.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetMovementState.restype = ctypes.c_bool

        self.windll.VCS_MoveWithVelocity.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_MoveWithVelocity.restype = ctypes.c_bool

        self.windll.VCS_HaltVelocityMovement.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_HaltVelocityMovement.restype = ctypes.c_bool

        self.windll.VCS_SetOperationMode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_int8,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_SetOperationMode.restype = ctypes.c_bool

        self.windll.VCS_GetVelocityIs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetVelocityIs.restype = ctypes.c_bool

        self.windll.VCS_GetVelocityIsAveraged.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetVelocityIsAveraged.restype = ctypes.c_bool

        self.windll.VCS_GetTargetVelocity.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetTargetVelocity.restype = ctypes.c_bool

        self.windll.VCS_GetCurrentMust.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_short),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetPositionProfile.restype = ctypes.c_bool

        self.windll.VCS_SetCurrentMust.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_short,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_SetCurrentMust.restype = ctypes.c_bool

        self.windll.VCS_GetPositionProfile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_GetPositionProfile.restype = ctypes.c_bool

        self.windll.VCS_SetPositionProfile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_SetPositionProfile.restype = ctypes.c_bool

        self.windll.VCS_GetErrorInfo.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint16]
        self.windll.VCS_GetErrorInfo.restype = ctypes.c_bool

        self.windll.VCS_ActivateHomingMode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_ActivateHomingMode.restype = ctypes.c_bool

        self.windll.VCS_FindHome.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_byte,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_FindHome.restype = ctypes.c_bool

        self.windll.VCS_DefinePosition.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_DefinePosition.restype = ctypes.c_bool

        self.windll.VCS_GetDeviceNameSelection.argtypes = [
            ctypes.c_bool,
            ctypes.c_char_p,
            ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_DefinePosition.restype = ctypes.c_bool

        self.windll.VCS_ImportParameter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_ImportParameter.restype = ctypes.c_bool

        self.windll.VCS_ExportParameter.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_bool,
            ctypes.c_bool,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.windll.VCS_ExportParameter.restype = ctypes.c_bool


if __name__ == "__main__":
    motor = EPOSMotor()
    motor.activate_profile_velocity_mode()
    motor.move_with_velocity(500)
