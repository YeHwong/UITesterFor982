#!/usr/bin/env python

import MegaSigLib as megasigLib


if __name__ == '__main__':
    obj_megasig = megasigLib.MegaSigLib()
    obj_megasig.load_dll_init()

    obj_megasig.get_usb_device_cnt()
    for each_dev_idx in range(obj_megasig.mega_dev_cnt):
        obj_megasig.get_usb_device_name(param_dev_idx=each_dev_idx)

    obj_megasig.mgs_init(param_dev_idx=0)

    obj_megasig.ai_channels()
    obj_megasig.ao_channels()

    # init params
    SampleRate = 16000
    NSampleCntPerChnn = 8000

    # create output task
    obj_megasig.create_ao_channel()
    # config setting
    obj_megasig.daq_timing(param_acq_type=megasigLib.AcqType.TypeAO.value,
                           param_db_expect_sample_rate=SampleRate,
                           param_n_sample_per_chnn=NSampleCntPerChnn)

    # start task
    obj_megasig.start_ao()

    # --------------------------------------------- Method A --------------------------------------------- #
    # parse wave file
    # obj_megasig.parse_data_wave(param_target_wave_path=r'.\wav\freq_swept_5s_2chnn_2width.wav')

    # write data here
    # obj_megasig.write_double_data(param_nlv=0,
    #                               param_nrow=4,
    #                               param_ncol=24000,
    #                               param_db_timeouts=5.0,
    #                               param_output_order=['L', 'R', 'L', 'R'])

    # --------------------------------------------- Method B --------------------------------------------- #
    # gen freq swept raw data
    freq_swept_nparray = megasigLib.gen_freq_swpet_cosine_wave(param_save_wav_path=r'.\wav\megasig_acq_d_track_12chnn_2width_15999rate.wav',
                                                               param_time_span=0.35152,
                                                               param_sample_rate=SampleRate,
                                                               param_freq_start=20.0,
                                                               param_freq_end=20000.0,
                                                               param_wav_chnn_num=2,
                                                               param_wav_sample_width=2)
    obj_megasig.lchann_data_array = freq_swept_nparray / megasigLib.DataCalValue
    obj_megasig.rchann_data_array = freq_swept_nparray / megasigLib.DataCalValue

    obj_megasig.write_double_data(param_nlv=0,
                                  param_nrow=2,
                                  param_ncol=NSampleCntPerChnn,
                                  param_db_timeouts=5.0,
                                  param_output_order=['L', 'R'])

    # stop task
    obj_megasig.stop_ao()

    obj_megasig.load_dll_deinit()
