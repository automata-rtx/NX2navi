#pragma once
#include <stdint.h>

// Reference the actual native recording part. Caller owns one reference even if the part has no texture uses.
typedef struct VkRecordingPartDisposition VkRecordingPartDisposition;
VkRecordingPartDisposition *vk_recording_part_disposition_capture(uint64_t fence);
int vk_recording_part_disposition_handed(const VkRecordingPartDisposition *);
void vk_recording_part_disposition_release(VkRecordingPartDisposition *);
