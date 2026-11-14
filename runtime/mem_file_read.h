#pragma once
#include "platform/platform.h"

// Keep guest destination pages writable until the positional read has completed.
int64_t mem_file_read_at(PlatFile *file, void *buf, uint32_t size, uint64_t offset);
