#ifndef   __MGS_USB_TOOLDLL_H__
#define   __MGS_USB_TOOLDLL_H__
#endif

#include<Windows.h>
#include "vector"

#define  EXPORT_DLL   extern "C" __declspec(dllexport)

/*!
 * Mgs_Start_Acquisition
 */
EXPORT_DLL int Mgs_Start_Acquisition(int nDeviceID,double Sample_Rate, double Acq_TimeS,int Acq_Channel, char Save_Wav_Path[], double Acq_Gain);

/*!
 * MgsStartAcquisition
 */
EXPORT_DLL int MgsStartAcquisition(int nDeviceID,double Sample_Rate, double Acq_TimeS,int Acq_Channel, char Save_Wav_Path[], double Acq_Gain);

/*!
 * Mgs_Stop_Acquisition
 */
EXPORT_DLL int Mgs_Stop_Acquisition(int nDeviceID);
/*!
 * Mgs_Init
 */
EXPORT_DLL int Mgs_Init(int nDeviceID);

/*!
 * MgsPlayWav
 */
EXPORT_DLL int Mgs_Play_Wav(int nDeviceID,char Wav_Play_Path[], double Sample_Rate,int Play_Channel, double Play_Level);

/*!
 * Mgs_Start_Play_Wav
 */
EXPORT_DLL int Mgs_Start_Play_Wav(int nDeviceID,char Wav_Play_Path[], double Sample_Rate, int Play_Channel, double Play_Level);
/*!
 * Mgs_Stop_Play_Wav
 */
EXPORT_DLL int  Mgs_Stop_Play_Wav(int nDeviceID);


EXPORT_DLL int GetUsbDeviceCount(int *pnCount);
EXPORT_DLL int GetUsbDeviceName(int nDeviceID,char *szOutputbuff);

EXPORT_DLL int DAQ_Timing(int nDeviceID,int nAcqType,double dbExpectSampleRate,int nSamplePerChannel,double *pdbActualSampleRate,int *pnActualSamplePerChannel);

EXPORT_DLL int Create_AO_Channel(int nDeviceID);
EXPORT_DLL int StartAO(int nDeviceID);
EXPORT_DLL int WriteDoubleData(int nDeviceID,int nLV,double **ppdbWriteData,int nRow,int nColum,double dbTimeOutS);
EXPORT_DLL void StopAO(int nDeviceID);
EXPORT_DLL int WaitAODone(int nDeviceID,double dbTimeOutS);

EXPORT_DLL int Create_AI_Channel(int nDeviceID);
EXPORT_DLL int StartAI(int nDeviceID);
EXPORT_DLL int ReadDoubleData(int nDeviceID,int nSamplePerChannel,int nLV,double **ppdbReadData,double dbTimeOutS);
EXPORT_DLL void StopAI(int nDeviceID);

EXPORT_DLL int AI_Channels(int nDeviceID);
EXPORT_DLL int AO_Channels(int nDeviceID);

EXPORT_DLL char* ParseErrCode(bool bEnglishMsg,int nUsbDAQErrCode);




