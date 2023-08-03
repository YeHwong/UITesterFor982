#!/usr/bin/env python

from ctypes import cdll
from ctypes import byref, POINTER
from ctypes import c_int, c_double, c_bool, c_char_p
from ctypes import create_string_buffer

from enum import unique, IntEnum
from lib.LogSet import Logger
import math
import numpy as np
import os
import wave

DataCalValue = 32767


@unique                             # unique  去除重复值
class AOChanSel(IntEnum):
    """
        音频输出通道的通道号

    """
    AO1_AO2 = 0x0,
    AO2_AO3 = 0x1,
    AO3_AO4 = 0x2,
    AO_Max = 0x3,


@unique
class AIChanSel(IntEnum):
    """
        音频输入通道的通道号

    """
    AI1_AI2 = 0x0,
    AI2_AI3 = 0x1,
    AI3_AI4 = 0x2,
    AI4_AI5 = 0x3,
    AI5_AI6 = 0x4,
    AI6_AI7 = 0x5,
    AI7_AI8 = 0x6,
    AI_Max = 0x7,


@unique
class AcqType(IntEnum):
    TypeAI = 0x00,
    TypeAO = 0x01,
    Type_Max = 0x02,


@unique
class ErrCode(IntEnum):
    Err_ok = 0,
    Err_fail = -1,


def megasig_freq_sweep_core(sweep_config, param_megasig):
    """
    MegaSig frequency sweep core interfacec.
    :param sweep_config: Target sweep config.
    :param param_megasig: Target megasig instance handler.
    """

    target_freq_start = sweep_config.sweep_freq_low
    target_freq_end = sweep_config.sweep_freq_high
    target_sample_rate = sweep_config.sweep_sample_rate
    target_play_timespan = sweep_config.sweep_audio_duration
    target_record_timespan = sweep_config.sweep_rec_duration
    target_sample_cnt_perchnn = math.ceil(target_sample_rate * target_play_timespan)

    if None is param_megasig:
        param_megasig = MegaSigLib()

    param_megasig.load_dll_init()

    param_megasig.get_usb_device_cnt()
    for _each_dev_idx in range(param_megasig.mega_dev_cnt):
        param_megasig.get_usb_device_name(param_dev_idx=_each_dev_idx)

    param_megasig.mgs_init(param_dev_idx=0)

    # create input task
    param_megasig.create_ai_channel()
    # config setting
    param_megasig.daq_timing(param_acq_type=AcqType.TypeAI.value,
                             param_db_expect_sample_rate=int(target_sample_rate),
                             param_n_sample_per_chnn=target_sample_cnt_perchnn)

    # start task
    param_megasig.start_ai()

    # ---------------------------- start output
    # create output task
    param_megasig.create_ao_channel()
    # config setting
    param_megasig.daq_timing(param_acq_type=AcqType.TypeAO.value,
                             param_db_expect_sample_rate=target_sample_rate,
                             param_n_sample_per_chnn=target_sample_cnt_perchnn)

    # start task
    param_megasig.start_ao()

    # gen freq swept raw data
    freq_swept_nparray = gen_freq_swpet_cosine_wave(param_save_wav_path=r'.\wav\freq_swept.wav',
                                                    param_time_span=target_play_timespan,
                                                    param_sample_rate=target_sample_rate,
                                                    param_freq_start=target_freq_start,
                                                    param_freq_end=target_freq_end,
                                                    param_wav_chnn_num=2,
                                                    param_wav_sample_width=2)
    param_megasig.lchann_data_array = freq_swept_nparray / DataCalValue
    param_megasig.rchann_data_array = freq_swept_nparray / DataCalValue

    param_megasig.write_double_data(param_nlv=0,
                                    param_nrow=4,
                                    param_ncol=target_sample_cnt_perchnn,
                                    param_db_timeouts=5.0,
                                    param_output_order=['L', 'R', 'L', 'R'])

    # stop task
    param_megasig.stop_ao()
    # ---------------------------- stop output

    # read data here
    param_megasig.read_double_data(param_n_sample_per_chnn=param_megasig.actual_sample_cnt,
                                   param_nlv=0,
                                   param_db_timeouts=5.0,
                                   param_acq_time_span=target_record_timespan + 0.4)
    # process data and save wave
    wave_data_2darray = param_megasig.process_pari_data_save_wave(param_save_wave_path='megasig_acq.wav',
                                                                  param_chann_pair=(1, 2))
    wav_data_sel_slice = slice(0, int(target_sample_rate * target_record_timespan))

    # stop task
    param_megasig.stop_ai()
    # --------------------------------------- stop acquicition

    param_megasig.load_dll_deinit()

    return wave_data_2darray[wav_data_sel_slice, :]


def gen_freq_swpet_cosine_wave(param_save_wav_path,
                               param_time_span,
                               param_sample_rate,
                               param_freq_start,
                               param_freq_end,
                               param_wav_chnn_num,
                               param_wav_sample_width):
    """
    Frequency-swept cosine generator, save to wav.

    :param param_save_wav_path: Target saved wave file path.
    :param param_time_span: Target audio timespan.
    :param param_sample_rate: Target sample rate.
    :param param_freq_start: Target frequency start.
    :param param_freq_end: Target frequency end.
    :param param_wav_chnn_num: Target wave channel number.
    :param param_wav_sample_width: Target wave sample width.
    """
    assert (param_wav_chnn_num in [1, 2])
    assert (param_wav_sample_width == 2)

    from scipy.signal import chirp

    t = np.arange(0, param_time_span, 1.0 / param_sample_rate)
    wave_data = chirp(t, param_freq_start, param_time_span, param_freq_end, method='logarithmic') * 10000
    wave_data = wave_data.astype(np.int16)

    if param_save_wav_path:
        # save to wave
        saved_file_dir = os.path.dirname(param_save_wav_path)
        saved_file_base_name = os.path.basename(param_save_wav_path)
        saved_filename, saved_fileext = os.path.splitext(saved_file_base_name)
        if r".wav" != saved_fileext:
            raise Exception(f"Invalid wave file extension: <{saved_file_base_name}>")
        param_save_wav_path = os.path.join(saved_file_dir,
                                           f"{saved_filename}_"
                                           f"{param_time_span}s_"
                                           f"{param_wav_chnn_num}chnn_"
                                           f"{param_wav_sample_width}width_"
                                           f"{param_sample_rate}rate.wav")

        try:
            wf = wave.open(param_save_wav_path, 'wb')
            wf.setnchannels(param_wav_chnn_num)
            wf.setsampwidth(param_wav_sample_width)
            wf.setframerate(param_sample_rate)

            # check channel
            if 1 == param_wav_chnn_num:
                # nparray.shape = (param_time_span * param_sample_rate, )   (c_int16)
                # len_wave_frames = 2 * param_time_span * param_sample_rate (bytes)
                target_wave_frames = wave_data.tostring()
            elif 2 == param_wav_chnn_num:
                # nparray.shape = (param_time_span * param_sample_rate, 2)  (c_int16)
                # len_wave_frames = 2 * param_time_span * param_sample_rate (bytes)
                wave_data_2dim = np.concatenate([wave_data, wave_data], axis=0)
                # reshape to 2-row
                wave_data_2dim = wave_data_2dim.reshape(2, wave_data.shape[0])
                # transpose itself to 2-col
                wave_data_2dim_t = wave_data_2dim.T
                # convert 2 1darray
                wave_data_1dim = wave_data_2dim_t.reshape((2 * wave_data.shape[0]))
                target_wave_frames = wave_data_1dim
            else:
                raise Exception(f"Invalid audio channel number: <{param_wav_chnn_num}>")

            wf.writeframes(target_wave_frames)
            wf.close()
            print(f"Save to wav: <{param_save_wav_path}>")
        except Exception as e_info:
            print(f"{e_info}")
            print(f"Failed to save wav: <{param_save_wav_path}>")

    return wave_data


class MegaSigLib:
    """
    MegaSig Library.
    """

    Cur_work_dir = os.path.dirname(__file__)
    Dll_file_path = os.path.join(Cur_work_dir, r"Mgs_USB_Tool.dll")

    wav_file_dir = os.path.join(Cur_work_dir, r".\wav")
    if not os.path.exists(wav_file_dir):
        os.makedirs(wav_file_dir)
    # Log_file_dir = os.path.join(Cur_work_dir, r"MegaSig_LOG")
    # if not os.path.exists(Log_file_dir):
    #     os.makedirs(Log_file_dir)

    def __init__(self):
        self.mega_sig_dll = None

        self.mega_dev_name = ''
        self.wav_save_chnn_num = 0
        self.wav_save_sample_width = 2
        self.mega_dev_cnt, self.mega_dev_idx = 0, 0
        self.chunk_read_time, self.chunk_write_time = 0, 0
        self.actual_sample_rate, self.actual_sample_cnt = 0, 0

        self.chnn_num_lst = [1, 2, 3, 4]
        self.total_chnns_2d_array_lst = list()
        self.output_4chnn_np_array_lst = list()

        self.lchann_data_array = np.array([], dtype=np.int16)
        self.rchann_data_array = np.array([], dtype=np.int16)

        # self.logger_mega = None
        # self.logger_fmt = None
        # self.logger_file_handler, self.logger_stream_handler = None, None
        # self.logger_init(param_logger_idx)
        self.mega_log = Logger(base_dir='./MegaSig_LOG', level='info').my_logger
        # 这里的路径要加./ 不然不是在当前目录下建文件夹，而是在c:/建文件夹
        self.ao_channel_cnt = 4
        self.ai_channel_cnt = 4

    def load_dll_init(self):
        """
        Load MegaSig DLL (based on C: __declspec).
        """
        os.chdir(MegaSigLib.Cur_work_dir)
        self.mega_sig_dll = cdll.LoadLibrary(MegaSigLib.Dll_file_path)
        self.mega_log.info(f"Init Dll <{MegaSigLib.Dll_file_path}> completes!")

    def load_dll_deinit(self):
        """
        Deinit dll load handler.
        """
        self.mega_sig_dll = None
        self.mega_log.info(f"Deinit Dll completes!")

    def get_usb_device_cnt(self):
        """
        Dll api: <GetUsbDeviceCount>
        :return: Error code.
        """
        target_cnt = c_int()
        dll_ret = self.mega_sig_dll.GetUsbDeviceCount(byref(target_cnt))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <GetUsbDeviceCount>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_dev_cnt = target_cnt.value
            self.mega_log.info(f"Detected available {self.mega_dev_cnt} usb device.")
        return dll_ret

    def get_usb_device_name(self, param_dev_idx=0):
        """
        Dll api: <GetUsbDeviceName>
        :param param_dev_idx: Target device index.
        :return: Error code.
        """
        assert (isinstance(param_dev_idx, int))
        assert (0 <= param_dev_idx < self.mega_dev_cnt)

        target_name_buf = create_string_buffer(256)
        dll_ret = self.mega_sig_dll.GetUsbDeviceName(c_int(param_dev_idx), target_name_buf)
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <GetUsbDeviceName> - param: devID={param_dev_idx}")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            target_name_bytes = target_name_buf.raw
            target_name_bytes = target_name_bytes.replace(b'\x00', b'')
            self.mega_dev_name = target_name_bytes.decode('utf-8')
            self.mega_log.info(f"   {param_dev_idx} >>> MegaSig device: '{self.mega_dev_name}'")
        return dll_ret

    def mgs_init(self, param_dev_idx=0):
        """
        Dll api: <Mgs_Init>
        :param param_dev_idx: Target device index.
        :return: Error code.
        """
        assert (isinstance(param_dev_idx, int))
        assert (0 <= param_dev_idx < self.mega_dev_cnt)

        dll_ret = self.mega_sig_dll.Mgs_Init(c_int(param_dev_idx))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Mgs_Init> - param: devID={param_dev_idx}")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_dev_idx = param_dev_idx
            self.mega_log.info(f"Init device index<{param_dev_idx}> successfully.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def mgs_play_wav_til_end(self,
                             param_wave_path,
                             param_sample_rate,
                             param_play_chnn,
                             param_play_lvl):
        """
        Dll api: <Mgs_Play_Wav> - 'api processed untill wave play to complete'
        :param param_wave_path: Target wave file path.
        :param param_sample_rate: Target sample rate.
        :param param_play_chnn: Target play channel index.
        :param param_play_lvl: Target play level percent.
        :return: Error code.
        """
        assert (isinstance(param_wave_path, str))
        assert (isinstance(param_sample_rate, int))
        assert (isinstance(param_play_chnn, int))
        assert (0 <= param_play_chnn < AOChanSel.AO_Max)
        assert (isinstance(param_play_lvl, float))

        dll_ret = self.mega_sig_dll.Mgs_Play_Wav(c_int(self.mega_dev_idx),
                                                 create_string_buffer(param_wave_path.encode('utf-8')),
                                                 c_double(param_sample_rate),
                                                 c_int(param_play_chnn),
                                                 c_double(param_play_lvl))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Mgs_Play_Wav>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega played wave completes.")
        self.mega_log.info(f" - param: devID={self.mega_dev_idx}")
        self.mega_log.info(f" - param: wavePath={param_wave_path}")
        self.mega_log.info(f" - param: sampleRate={param_sample_rate}")
        self.mega_log.info(f" - param: playChnn={param_play_chnn}")
        self.mega_log.info(f" - param: playLvl={param_play_lvl}")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def mgs_start_play_wav(self,
                           param_wave_path,
                           param_sample_rate,
                           param_play_chnn,
                           param_play_lvl):
        """
        Dll api: <Mgs_Start_Play_Wav> - 'api processed in thread'
        :param param_wave_path: Target wave file path.
        :param param_sample_rate: Target sample rate.
        :param param_play_chnn: Target play channel index.
        :param param_play_lvl: Target play level percent.
        :return: Error code.
        """
        assert (isinstance(param_wave_path, str))
        assert (isinstance(param_sample_rate, int))
        assert (isinstance(param_play_chnn, int))
        assert (0 <= param_play_chnn < AOChanSel.AO_Max)
        assert (isinstance(param_play_lvl, float))

        dll_ret = self.mega_sig_dll.Mgs_Start_Play_Wav(c_int(self.mega_dev_idx),
                                                       create_string_buffer(param_wave_path.encode('utf-8')),
                                                       c_double(param_sample_rate),
                                                       c_int(param_play_chnn),
                                                       c_double(param_play_lvl))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Mgs_Start_Play_Wav>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega start to play wave completes.")
        self.mega_log.info(f" - param: devID={self.mega_dev_idx}")
        self.mega_log.info(f" - param: wavePath={param_wave_path}")
        self.mega_log.info(f" - param: sampleRate={param_sample_rate}")
        self.mega_log.info(f" - param: playChnn={param_play_chnn}")
        self.mega_log.info(f" - param: playLvl={param_play_lvl}")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def mgs_stop_play_wav(self):
        """
        Dll api: <Mgs_Stop_Play_Wav> - 'Execution thread'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.Mgs_Stop_Play_Wav(c_int(self.mega_dev_idx))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Mgs_Stop_Play_Wav>")
        else:
            self.mega_log.info(f"Mega stop to play wave completes.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def mgs_start_acquisition(self,
                              param_sample_rate,
                              param_acq_times,
                              param_acq_chnn,
                              param_save_wave_path,
                              param_acq_gain):
        """
        Dll api: <Mgs_Start_Acquisition> - 'Execution thread'
        :param param_sample_rate: Target sample rate.
        :param param_acq_times: Target acquire time seconds.
        :param param_acq_chnn: Target acquire channel numner.
        :param param_save_wave_path: Target wave file save path.
        :param param_acq_gain: Target acquire gain.
        :return: Error code.
        """
        assert (isinstance(param_sample_rate, int))
        assert (isinstance(param_acq_times, float))
        assert (isinstance(param_acq_chnn, int))
        assert (0 <= param_acq_chnn < AIChanSel.AI_Max)
        assert (isinstance(param_save_wave_path, str))
        assert (isinstance(param_acq_gain, float))

        dll_ret = self.mega_sig_dll.Mgs_Start_Acquisition(c_int(self.mega_dev_idx),
                                                          c_double(param_sample_rate),
                                                          c_double(param_acq_times),
                                                          c_int(param_acq_chnn),
                                                          create_string_buffer(param_save_wave_path.encode('utf-8')),
                                                          c_double(param_acq_gain))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Mgs_Start_Acquisition>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega acquisition started.")
        self.mega_log.info(f" - param: devID={self.mega_dev_idx}")
        self.mega_log.info(f" - param: sampleRate={param_sample_rate}")
        self.mega_log.info(f" - param: acqTimes={param_acq_times}")
        self.mega_log.info(f" - param: acqChnn={param_acq_chnn}")
        self.mega_log.info(f" - param: saveWavePath={param_save_wave_path}")
        self.mega_log.info(f" - param: acqGain={param_acq_gain}")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def mgs_stop_acquisition(self):
        """
        Dll api: <Mgs_Stop_Acquisition> - 'Execution thread'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.Mgs_Stop_Acquisition(c_int(self.mega_dev_idx))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Mgs_Stop_Acquisition>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega acquisition stoped.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def create_ai_channel(self):
        """
        Dll api: <Create_AI_Channel> - 'create data acquisition task by device id'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.Create_AI_Channel(c_int(self.mega_dev_idx))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Create_AI_Channel>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega data acquisition task created.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def create_ao_channel(self):
        """
        Dll api: <Create_AO_Channel> - 'create data output task by device id'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.Create_AO_Channel(c_int(self.mega_dev_idx))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <Create_AO_Channel>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega data acquisition task created.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def daq_timing(self,
                   param_acq_type,
                   param_db_expect_sample_rate,
                   param_n_sample_per_chnn):
        """
        Dll api: <DAQ_Timing> - 'configurate setting by device id'
        :param param_acq_type: Target acq type.
        :param param_db_expect_sample_rate: Target sameple rate which device expected.
        :param param_n_sample_per_chnn: Target sample counter per channel.
        :return: Error code.
        """
        assert (isinstance(param_acq_type, int))
        assert (0 <= param_acq_type < AcqType.Type_Max)
        assert (isinstance(param_db_expect_sample_rate, int))
        assert (isinstance(param_n_sample_per_chnn, int))

        target_actual_sample_rate = c_double()
        target_actual_sample_count = c_int()

        dll_ret = self.mega_sig_dll.DAQ_Timing(c_int(self.mega_dev_idx),
                                               c_int(param_acq_type),
                                               c_double(param_db_expect_sample_rate),
                                               c_int(param_n_sample_per_chnn),
                                               byref(target_actual_sample_rate),
                                               byref(target_actual_sample_count))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <DAQ_Timing>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega adq timing setting completes.")
        self.mega_log.info(f" - param: devID={self.mega_dev_idx}")
        self.mega_log.info(f" - param: acqType={param_acq_type}")
        self.mega_log.info(f" - param: sampleRate={param_db_expect_sample_rate}")
        self.mega_log.info(f" - param: sampleCount={param_n_sample_per_chnn}")

        self.actual_sample_rate = int(target_actual_sample_rate.value)
        self.actual_sample_cnt = target_actual_sample_count.value

        self.mega_log.info(f" - return: sampleRate={target_actual_sample_rate.value}")
        self.mega_log.info(f" - return: sampleCount={target_actual_sample_count.value}")
        self.mega_log.info(f"-*-" * 20)
        return self.actual_sample_rate, self.actual_sample_cnt

    def start_ai(self):
        """
        Dll api: <StartAI> - 'start data acquisition task by device id'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.StartAI(c_int(self.mega_dev_idx))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <StartAI>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega data acquisition task started.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def stop_ai(self):
        """
        Dll api: <StopAI> - 'stop data acquisition task by device id'
        """
        self.mega_sig_dll.StopAI(c_int(self.mega_dev_idx))

    def start_ao(self):
        """
        Dll api: <StartAO> - 'start data output task by device id'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.StartAO(c_int(self.mega_dev_idx))
        if dll_ret:
            self.mega_log.error(f"Dll api failed: <StartAO>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega data output task started.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def stop_ao(self):
        """
        Dll api: <StopAO> - 'stop data output task by device id'
        """
        self.mega_sig_dll.StopAO(c_int(self.mega_dev_idx))

    def wait_ao_done(self, param_db_timeouts):
        """
        Dll api: <WaitAODone> - 'wait data output task finish by device id'
        :param param_db_timeouts: Target read data timeout seconds.
        :return: Error code.
        """
        assert (isinstance(param_db_timeouts, float))

        dll_ret = self.mega_sig_dll.WaitAODone(c_int(self.mega_dev_idx),
                                               c_double(param_db_timeouts))

        if dll_ret:
            self.mega_log.error(f"Dll api failed: <WaitAODone>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Mega wait data output task finished.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def ai_channels(self):
        """
        Dll api: <AI_Channels> - 'get data acquisition channel count by device id'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.AI_Channels(c_int(self.mega_dev_idx))
        self.ai_channel_cnt = dll_ret
        if dll_ret < 0:
            self.mega_log.error(f"Dll api failed: <AI_Channels>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Query mega data acquisition channel: <{dll_ret}>.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def ao_channels(self):
        """
        Dll api: <AO_Channels> - 'get data output channel count by device id'
        :return: Error code.
        """
        dll_ret = self.mega_sig_dll.AO_Channels(c_int(self.mega_dev_idx))
        self.ao_channel_cnt = dll_ret
        if dll_ret < 0:
            self.mega_log.error(f"Dll api failed: <AO_Channels>")
            self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
        else:
            self.mega_log.info(f"Query mega data output channel: <{dll_ret}>.")
        self.mega_log.info(f"-*-" * 20)
        return dll_ret

    def read_double_data(self,
                         param_n_sample_per_chnn,
                         param_nlv,
                         param_db_timeouts,
                         param_acq_time_span):
        """
        Dll api: <ReadDoubleData> - 'read all channel acquisition data'
        :param param_n_sample_per_chnn: Target sample count per channel.
        :param param_nlv: 0 needed.
        :param param_db_timeouts: Target read data timeout seconds.
        :param param_acq_time_span: Target read data timespan seconds.
        :return: Error code.
        """
        assert (isinstance(param_n_sample_per_chnn, int))
        assert (param_nlv == 0)
        assert (isinstance(param_db_timeouts, float))
        assert (isinstance(param_acq_time_span, float))

        self.chunk_read_time = int(param_acq_time_span / (param_n_sample_per_chnn / self.actual_sample_rate))

        self.total_chnns_2d_array_lst = list()
        target_total_chnns_2d_array_ptr_lst = list()
        for each_cnt in range(self.chunk_read_time):
            # 2d array: ncol=param_n_sample_per_chnn, nrow=DataChnnCnt
            target_per_chnns_2d_array = (c_double * param_n_sample_per_chnn * self.ai_channel_cnt)()
            target_per_chnns_2d_array_ptr = (POINTER(c_double) * self.ai_channel_cnt)(*target_per_chnns_2d_array)

            self.total_chnns_2d_array_lst.append(target_per_chnns_2d_array)
            target_total_chnns_2d_array_ptr_lst.append(target_per_chnns_2d_array_ptr)

        dll_ret = None
        for each_idx, each_chnns_2d_array_pty in enumerate(target_total_chnns_2d_array_ptr_lst):
            dll_ret = self.mega_sig_dll.ReadDoubleData(c_int(self.mega_dev_idx),
                                                       c_int(param_n_sample_per_chnn),
                                                       c_int(param_nlv),
                                                       each_chnns_2d_array_pty,
                                                       c_double(param_db_timeouts))
            if dll_ret:
                self.mega_log.error(f"Dll api failed: <ReadDoubleData>")
                self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
                break
            else:
                print(np.array(self.total_chnns_2d_array_lst))
                self.mega_log.info(f"Mega read double data {each_idx + 1}time completes: {dll_ret}.")
            self.mega_log.debug(f" - param: devID={self.mega_dev_idx}")
            self.mega_log.debug(f" - param: sampleCount={param_n_sample_per_chnn}")
            self.mega_log.debug(f" - param: nLv={param_nlv}")
            self.mega_log.debug(f" - param: timeOuts={param_db_timeouts}")
            self.mega_log.info(f"-*-" * 20)

        return dll_ret

    def read_single_data(self,
                         param_n_sample_per_chnn,
                         param_nlv,
                         param_db_timeouts,
                         param_acq_time_span):
        """
        Dll api: <ReadSingleData> - 'read all channel acquisition data'
                    读取单通道 数据
        :param param_n_sample_per_chnn: Target sample count per channel.
        :param param_nlv: 0 needed.
        :param param_db_timeouts: Target read data timeout seconds.
        :param param_acq_time_span: Target read data timespan seconds.
        :return: Error code.
        """
        assert (isinstance(param_n_sample_per_chnn, int))
        assert (param_nlv == 0)
        assert (isinstance(param_db_timeouts, float))
        assert (isinstance(param_acq_time_span, float))

        self.chunk_read_time = int(param_acq_time_span / (param_n_sample_per_chnn / self.actual_sample_rate))

        self.total_chnns_2d_array_lst = list()
        target_total_chnns_2d_array_ptr_lst = list()
        for each_cnt in range(self.chunk_read_time):
            # 2d array: ncol=param_n_sample_per_chnn, nrow=DataChnnCnt
            target_per_chnns_2d_array = (c_double * param_n_sample_per_chnn * self.ai_channel_cnt)()
            target_per_chnns_2d_array_ptr = (POINTER(c_double) * self.ai_channel_cnt)(*target_per_chnns_2d_array)

            self.total_chnns_2d_array_lst.append(target_per_chnns_2d_array)
            target_total_chnns_2d_array_ptr_lst.append(target_per_chnns_2d_array_ptr)

        dll_ret = None
        for each_idx, each_chnns_2d_array_pty in enumerate(target_total_chnns_2d_array_ptr_lst):
            dll_ret = self.mega_sig_dll.ReadDoubleData(c_int(self.mega_dev_idx),
                                                       c_int(param_n_sample_per_chnn),
                                                       c_int(param_nlv),
                                                       each_chnns_2d_array_pty,
                                                       c_double(param_db_timeouts))
            if dll_ret:
                self.mega_log.error(f"Dll api filed: <ReadDoubleData>")
                self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
                break
            else:
                print(np.array(self.total_chnns_2d_array_lst))
                self.mega_log.info(f"Mega read double data {each_idx + 1}time completes: {dll_ret}.")
            self.mega_log.debug(f" - param: devID={self.mega_dev_idx}")
            self.mega_log.debug(f" - param: sampleCount={param_n_sample_per_chnn}")
            self.mega_log.debug(f" - param: nLv={param_nlv}")
            self.mega_log.debug(f" - param: timeOuts={param_db_timeouts}")
            self.mega_log.info(f"-*-" * 20)

        return dll_ret

    def process_pari_data_save_wave(self,
                                    param_save_wave_path,
                                    param_chann_pair):
        """
        Process acquisition pair data and save them to wave file.
            处理采集的一对数据 然后保存到wav文件中
        :param param_save_wave_path: Target saved wave file path.    保存wav文件的目标路径
        :param param_chann_pair: Target channel pair tuple.     目标通道号的元组
        :return: Error code.    返回的错误码
        """
        assert (isinstance(param_chann_pair, (list, tuple)))    # 检查param_chann_pair类似是否为元组
        assert (0 < min(param_chann_pair) <= AIChanSel.AI_Max + 1)  # 检查param_chann_pair内的数值是否超区
        assert (0 < max(param_chann_pair) <= AIChanSel.AI_Max + 1)  # 检查param_chann_pair内的数值是否超区

        target_chnn_pair_lst = list(param_chann_pair)   # 将元组转为list存储
        target_chnn_pair_lst.sort() # 对类表内容进行排序时
        target_chnn_info = ''.join([str(i) for i in target_chnn_pair_lst])  # 合并列表为字符

        target_chnn_idx_lst = [i - 1 for i in target_chnn_pair_lst]
        self.wav_save_chnn_num = len(target_chnn_idx_lst)
        if 1 == self.wav_save_chnn_num:
            target_chnn_pair_slice = slice(target_chnn_idx_lst[0],
                                           target_chnn_idx_lst[0] + 1,
                                           1)
            target_track_info = 's_track'
        elif 2 == self.wav_save_chnn_num:
            target_chnn_pair_slice = slice(target_chnn_idx_lst[0],
                                           target_chnn_idx_lst[1] + 1,
                                           target_chnn_idx_lst[1] - target_chnn_idx_lst[0])
            target_track_info = 'd_track'
        else:
            raise Exception(f"Invalid selected data channel lst: <{param_chann_pair}>")

        self.mega_log.info(f"Selected data channel: {self.chnn_num_lst[target_chnn_pair_slice]}")

        tmp_array_data_lst = list()
        for each_chnns_2d_array in self.total_chnns_2d_array_lst:
            tmp_array_data_lst.append(np.array(each_chnns_2d_array))
        concat_array_data = np.concatenate(tmp_array_data_lst, axis=1)
        self.mega_log.info(f"Concatenate {self.chunk_read_time} read data arrays to shape: {concat_array_data.shape}")

        # transpose array here
        concat_array_data_t = np.transpose(concat_array_data)
        # select cols here
        concat_array_data_t = concat_array_data_t[:, target_chnn_pair_slice]
        # convert 2darray to 1darray
        data_1d_array = concat_array_data_t.reshape((concat_array_data_t.shape[0] * concat_array_data_t.shape[1]))
        # ?
        data_1d_array = data_1d_array * DataCalValue
        # convert double to c_int16 (size still same!!!)
        data_1d_array = data_1d_array.astype(np.int16)
        # convert to 2 int8
        # data_1d_array.dtype = np.int8
        # convert to bytes
        data_1d_array_bytes = data_1d_array.tostring()

        # save to wave
        saved_file_dir = os.path.dirname(param_save_wave_path)
        saved_file_base_name = os.path.basename(param_save_wave_path)
        saved_filename, saved_fileext = os.path.splitext(saved_file_base_name)
        if r".wav" != saved_fileext:
            raise Exception(f"Invalid wave file extension: <{saved_file_base_name}>")
        param_save_wave_path = os.path.join(saved_file_dir,
                                            f"{saved_filename}_"
                                            f"{target_track_info}_"
                                            f"{target_chnn_info}chnn_"
                                            f"{self.wav_save_sample_width}width_"
                                            f"{self.actual_sample_rate}rate.wav")

        try:
            wf = wave.open(param_save_wave_path, 'wb')  # 打开wav文件创建一个音频对象wf，开始写WAV文件
            wf.setnchannels(self.wav_save_chnn_num)  # 配置声道数
            wf.setsampwidth(self.wav_save_sample_width)  # 配置量化位数
            wf.setframerate(self.actual_sample_rate)  # 配置采样率
            wf.writeframes(data_1d_array_bytes)  # 转换为二进制数据写入文件
            wf.close()

            self.mega_log.info(f"Save wav setChannel={self.wav_save_chnn_num}, "
                             f"setSamRate={self.actual_sample_rate}, "
                             f"setSamWidth={self.wav_save_sample_width}")
            self.mega_log.info(f"Save to wave file <{param_save_wave_path}> completes.")
        except Exception as e_info:
            self.mega_log.error(f"Failed to save wave file <{param_save_wave_path}>")
            self.mega_log.error(str(e_info))

        return concat_array_data_t

    def parse_data_wave(self, param_target_wave_path):
        """
        Parse wave file into 4 channels data array lst.
        :param param_target_wave_path: Target parsed wave file.
        :return: Error code.
        """
        if not os.path.exists(param_target_wave_path):
            self.mega_log.error(f"Cannot find wave file: <{param_target_wave_path}>")
            return ErrCode.Err_fail.value

        wf = wave.open(param_target_wave_path, 'rb')
        wave_params = wf.getparams()
        wav_nchannels, wav_sampwidth, wav_framerate, wav_nframes = wave_params[:4]
        assert (0 < wav_nchannels <= 2)
        self.mega_log.info(f"Get wave <{param_target_wave_path}> params:")
        self.mega_log.info(f" - nchannels: {wav_nchannels}")
        self.mega_log.info(f" - sampwidth: {wav_sampwidth}")
        self.mega_log.info(f" - framerate: {wav_framerate}")
        self.mega_log.info(f" - nframes: {wav_nframes}")

        wave_data_bytes = wf.readframes(wav_nframes)

        wf.close()

        """
        通过fromstring函数将字符串转换为数组，通过其参数dtype指定转换后的数据格式，由于我们的声音格式是以两个字节表示一个取
        样值，因此采用short数据类型转换。现在我们得到的wave_data是一个一维的short类型的数组，但是因为我们的声音文件是双声
        道的，因此它由左右两个声道的取样交替构成：LRLRLRLR....LR(L表示左声道的取样值，R表示右声道取样值)
        """

        target_wave_data = np.frombuffer(wave_data_bytes, dtype=np.int16)
        target_wave_data = target_wave_data / DataCalValue
        target_wave_data.shape = (-1, wav_nchannels)
        target_wave_data = np.transpose(target_wave_data)

        # plt_time = np.arange(0, wav_nframes) * (1.0 / wav_framerate)

        # clr data array here
        tmp_lchann_data_array = (c_double * wav_nframes)()
        self.lchann_data_array = np.array(tmp_lchann_data_array)

        tmp_rchann_data_array = (c_double * wav_nframes)()
        self.rchann_data_array = np.array(tmp_rchann_data_array)

        for each_chnn_idx in range(wav_nchannels):
            if 0 == each_chnn_idx:
                self.lchann_data_array = target_wave_data[0]
            elif 1 == each_chnn_idx:
                self.rchann_data_array = target_wave_data[1]

        return ErrCode.Err_ok.value

    def write_double_data(self,
                          param_nlv,
                          param_nrow,
                          param_ncol,
                          param_db_timeouts,
                          param_output_order):
        """
        Dll api: <WriteDoubleData> - 'write all channel acquisition data into wave file'
        :param param_nlv: 0 needed.
        :param param_nrow: Target row count.
        :param param_ncol: Target column count.
        :param param_db_timeouts: Target write data timeout seconds.
        :param param_output_order: Target output data order, ['L', 'R', 'L', 'R'].
        :return: Error code.
        """
        assert (param_nlv == 0)
        assert (isinstance(param_nrow, int))
        assert (0 < param_nrow <= self.ao_channel_cnt)
        assert (isinstance(param_ncol, int))
        assert (isinstance(param_db_timeouts, float))
        assert (isinstance(param_output_order, (list, tuple)))
        assert (self.ao_channel_cnt == len(param_output_order))

        self.output_4chnn_np_array_lst = list()
        self.mega_log.debug(f"lchann_data_array: <{self.lchann_data_array}>")
        self.mega_log.debug(f"rchann_data_array: <{self.rchann_data_array}>")
        # re-order output
        for each_order in param_output_order:
            if each_order.lower().find('l') >= 0:
                self.output_4chnn_np_array_lst.append(self.lchann_data_array)
            elif each_order.lower().find('r') >= 0:
                self.output_4chnn_np_array_lst.append(self.rchann_data_array)
            else:
                raise Exception(f"Invalid output order lst: <{param_output_order}>")

        # check np array size
        check_array_size_set = set()
        for each_np_array in self.output_4chnn_np_array_lst:
            check_array_size_set.add(each_np_array.size)
        if len(check_array_size_set) != 1:
            self.mega_log.error(f"Mismatched numpy array size: <{len(check_array_size_set)}> of 4 channels.")
            return ErrCode.Err_fail.value

        total_data_len = list(check_array_size_set)[0]
        assert (param_ncol <= total_data_len)

        self.chunk_write_time = total_data_len // param_ncol
        if total_data_len % param_ncol:
            self.chunk_write_time += 1
        self.mega_log.info(f"Write data auto separated into {self.chunk_write_time} data chunks.")

        target_total_chnns_2d_array_ptr_lst = list()
        for each_cnt in range(self.chunk_write_time):
            # 2d array: ncol=param_n_sample_per_chnn, nrow=DataChnnCnt
            target_per_chnns_2d_array = (c_double * param_ncol * self.ao_channel_cnt)()

            # assign value into array
            for each_chnn in range(self.ao_channel_cnt):
                target_per_chnns_2d_array[each_chnn] = (c_double * param_ncol)(
                    *self.output_4chnn_np_array_lst[each_chnn][each_cnt * param_ncol: (each_cnt + 1) * param_ncol]
                )

            target_per_chnns_2d_array_ptr = (POINTER(c_double) * self.ao_channel_cnt)(*target_per_chnns_2d_array)
            target_total_chnns_2d_array_ptr_lst.append(target_per_chnns_2d_array_ptr)

        for each_idx, each_chnns_2d_array_pty in enumerate(target_total_chnns_2d_array_ptr_lst):
            dll_ret = self.mega_sig_dll.WriteDoubleData(c_int(self.mega_dev_idx),
                                                        c_int(param_nlv),
                                                        each_chnns_2d_array_pty,
                                                        c_int(param_nrow),
                                                        c_int(param_ncol),
                                                        c_double(param_db_timeouts))
            if dll_ret:
                self.mega_log.error(f"Dll api failed: <WriteDoubleData>")
                self.mega_log.error(f"Exception: <{self.parse_err_code(1, dll_ret)}>")
                break
            else:
                self.mega_log.info(f"Mega write double data {each_idx + 1}time completes: {dll_ret}.")
            self.mega_log.debug(f" - param: devID={self.mega_dev_idx}")
            self.mega_log.debug(f" - param: nLv={param_nlv}")
            self.mega_log.debug(f" - param: nrow={param_nrow}, ncol={param_ncol}")
            self.mega_log.debug(f" - param: timeOuts={param_db_timeouts}")
            self.mega_log.info(f"-*-" * 20)

        # wait all data send out
        dll_ret = self.wait_ao_done(param_db_timeouts)

        return dll_ret

    def parse_err_code(self, param_english_en, param_err_code):
        """
        Parse error code info.
        :param param_english_en: Target enable of english info output.
        :param param_err_code: Target error code.
        :return: Error info.
        """
        assert (param_english_en in [0, 1])
        assert (isinstance(param_err_code, int))

        self.mega_sig_dll.ParseErrCode.restype = c_char_p
        dll_ret = self.mega_sig_dll.ParseErrCode(c_bool(param_english_en),
                                                 c_int(param_err_code))
        err_info_str = dll_ret.decode('utf-8')

        return err_info_str


if __name__ == '__main__':
    obj_megasig = MegaSigLib()
    obj_megasig.load_dll_init()

    obj_megasig.get_usb_device_cnt()
    for each_dev_idx in range(obj_megasig.mega_dev_cnt):
        obj_megasig.get_usb_device_name(param_dev_idx=each_dev_idx)

    obj_megasig.mgs_init()
    obj_megasig.mgs_play_wav_til_end(param_wave_path=r"wav\megasig_acq_d_track_12chnn_2width_9600rate.wav",
                                     param_sample_rate=9600,
                                     param_play_chnn=AOChanSel.AO1_AO2.value,
                                     param_play_lvl=0.8)

    obj_megasig.mgs_start_play_wav(param_wave_path=r"wav\megasig_acq_d_track_12chnn_2width_9600rate.wav",
                                   param_sample_rate=9600,
                                   param_play_chnn=AOChanSel.AO1_AO2.value,
                                   param_play_lvl=0.8)

    print("---------------------timeout to stop play wave here---------------------")
    obj_megasig.mgs_stop_play_wav()

    # obj_megasig.mgs_start_acquisition(param_sample_rate=48000,
    #                                   param_acq_times=3.0,
    #                                   param_acq_chnn=0,
    #                                   param_save_wave_path=r".\wav\save_wave.wav",
    #                                   param_acq_gain=1.0)
    # time.sleep(5)
    # obj_megasig.mgs_stop_acquisition()

    obj_megasig.load_dll_deinit()
