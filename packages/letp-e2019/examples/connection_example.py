from letp_e2019.versions.e2019s import E2019S
from letp_e2019.versions.e2019b import E2019B
from letp_e2019.versions.e2019p import E2019P
from letp_e2019.versions.e2019q import E2019Q

available_e201 = {"P": E2019P, "B": E2019B, "S": E2019S, "Q": E2019Q}

e201_v = "B"
e2019 = available_e201.get(e201_v)("COM4")

print("Version:", e2019.get_version())
print("Build number:", e2019.get_build_number())
print("Supply:", e2019.get_supply())

e2019.set_read_command("ssi")

e2019.set_word_width(21)
e2019.set_clock_frequency(560)
print(e2019.read_clock_frequency())
raw_pos = e2019.read_position()
raw_pos = bytes.fromhex(raw_pos)
pos = int.from_bytes(raw_pos)
print(pos)
e2019.close()
