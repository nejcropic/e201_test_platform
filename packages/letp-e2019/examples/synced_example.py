from letp_e2019.e2019_synced import E2019Synced

config = {
    "e201_dut": {
        "comport": "COM81",
        "type": "E2019S",
    },
    "e201_ref": {
        "comport": "COM21",
        "type": "E2019Q",
    },
}

synced_sampler = E2019Synced(config)
synced_sampler.enable_synced_sampling()
print(synced_sampler.read_position())

synced_sampler.disable_synced_sampling()
synced_sampler.close_connection()
