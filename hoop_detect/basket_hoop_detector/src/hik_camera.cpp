#include "basket_hoop_detector/hik_camera.hpp"

#include <cstring>
#include <iostream>

namespace basket_hoop_detector
{
HikCamera::HikCamera() {}

HikCamera::~HikCamera() { close(); }

bool HikCamera::open()
{
  int ret = MV_CC_Initialize();
  if (ret != MV_OK)
  {
    std::cerr << "[HikCamera] Initialize SDK fail! ret: 0x" << std::hex << ret << std::endl;
    return false;
  }

  MV_CC_DEVICE_INFO_LIST device_list;
  memset(&device_list, 0, sizeof(device_list));
  ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
  if (ret != MV_OK || device_list.nDeviceNum == 0)
  {
    std::cerr << "[HikCamera] Enum devices fail or none found! ret: 0x" << std::hex << ret << std::endl;
    return false;
  }

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
  if (ret != MV_OK)
  {
    std::cerr << "[HikCamera] Create handle fail! ret: 0x" << std::hex << ret << std::endl;
    return false;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK)
  {
    std::cerr << "[HikCamera] Open device fail! ret: 0x" << std::hex << ret << std::endl;
    return false;
  }

  MV_CC_SetEnumValue(handle_, "ExposureAuto", 0);
  MV_CC_SetFloatValue(handle_, "ExposureTime", 3000.0f);
  MV_CC_SetEnumValue(handle_, "GainAuto", 0);
  MV_CC_SetFloatValue(handle_, "Gain", 23.9f);
  MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK)
  {
    std::cerr << "[HikCamera] Start grabbing fail! ret: 0x" << std::hex << ret << std::endl;
    return false;
  }
  std::cout << "[HikCamera] Camera opened successfully." << std::endl;
  return true;
}

bool HikCamera::getFrame(cv::Mat &frame)
{
  if (!handle_)
    return false;

  MV_FRAME_OUT frame_info;
  memset(&frame_info, 0, sizeof(frame_info));
  int ret = MV_CC_GetImageBuffer(handle_, &frame_info, 1000);
  if (ret == MV_OK)
  {
    cv::Mat img(frame_info.stFrameInfo.nHeight, frame_info.stFrameInfo.nWidth, CV_8UC3, frame_info.pBufAddr);
    img.copyTo(frame);
    MV_CC_FreeImageBuffer(handle_, &frame_info);
    return true;
  }
  return false;
}

void HikCamera::close()
{
  if (handle_)
  {
    MV_CC_StopGrabbing(handle_);
    MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
  }
  MV_CC_Finalize();
}

}  // namespace basket_hoop_detector 