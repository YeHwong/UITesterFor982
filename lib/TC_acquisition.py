#!/usr/bin/env python

import lib.MegaSigLib as megasigLib

if __name__ == '__main__':
    obj_megasig = megasigLib.MegaSigLib()
    obj_megasig.load_dll_init()

    obj_megasig.get_usb_device_cnt()
    for each_dev_idx in range(obj_megasig.mega_dev_cnt):
        obj_megasig.get_usb_device_name(param_dev_idx=each_dev_idx)

    obj_megasig.mgs_init(param_dev_idx=0)

    obj_megasig.ai_channels()
    obj_megasig.ao_channels()

    # create input task
    obj_megasig.create_ai_channel()
    # config setting
    obj_megasig.daq_timing(param_acq_type=megasigLib.AcqType.TypeAI.value,
                           param_db_expect_sample_rate=16000,
                           param_n_sample_per_chnn=16000)

    # start task
    obj_megasig.start_ai()

    # read data here
    obj_megasig.read_double_data(param_n_sample_per_chnn=obj_megasig.actual_sample_cnt,
                                 param_nlv=0,
                                 param_db_timeouts=5.0,
                                 param_acq_time_span=10.0)
    # process data and save wave
    obj_megasig.process_pari_data_save_wave(param_save_wave_path=r'.\wav\megasig_acq.wav',
                                            param_chann_pair=(1,))
    # obj_megasig.process_pari_data_save_wave(param_save_wave_path=r'.\wav\megasig_acq.wav',
    #                                        param_chann_pair=(3, 4))

    # stop task
    obj_megasig.stop_ai()

    obj_megasig.load_dll_deinit()
