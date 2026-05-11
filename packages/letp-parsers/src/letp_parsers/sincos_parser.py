"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      sincos_parser.py
brief:     Class for parsing sinus and cosinus signals acquired with ADC on Evalyn.
author(s): Nejc Ropič
date:      24.4.2026

details:   Provides parsing of eight adc probes from Evalyn. Also supports trimming of signals.
           Author of trimming algorithm was taken from Rok Kranjc.

"""

from letp_parsers.base_dut_parser import BaseDUTParser
import numpy as np


class SinCosParser(BaseDUTParser):
    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, dut_frame: bytes | list, dut_bytes):
        sine = dut_frame[self.dut_settings["sin_adc"] - 1]
        cosine = dut_frame[self.dut_settings["cos_adc"] - 1]
        ri = 0
        if self.dut_settings["ri_adc"] != 0:
            ri = dut_frame[self.dut_settings["ri_adc"] - 1]

        if not self.dut_settings["calibrate_signals"]:
            singleturn_position = np.arctan2(
                sine - self.dut_settings["se_offset"], cosine - self.dut_settings["se_offset"]
            )

        else:
            # Correcting amplitude by always raising it
            if self.dut_settings["sine_amplitude"] > self.dut_settings["cosine_amplitude"]:
                sine_amp_fact = 1
                cosine_amp_fact = self.dut_settings["sine_amplitude"] / self.dut_settings["cosine_amplitude"]
            else:
                cosine_amp_fact = 1
                sine_amp_fact = self.dut_settings["cosine_amplitude"] / self.dut_settings["sine_amplitude"]

            # Applying phase offset
            faz_fact = (self.dut_settings["phase_offset"] - 90) * np.pi / 180

            # Trimming algorithm
            timmed_sine = sine_amp_fact * (sine - self.dut_settings["sine_offset"])
            timmed_cosine = (
                cosine_amp_fact * (cosine - self.dut_settings["cosine_offset"]) + timmed_sine * np.sin(faz_fact)
            ) / np.cos(faz_fact)

            singleturn_position = np.arctan2(timmed_sine, timmed_cosine)

        return {
            "Position": singleturn_position,
            "Sine": sine,
            "Cosine": cosine,
            "Ri": ri,
            "CRC": False,
        }
