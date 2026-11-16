#pragma once
#include "cpu.h"

void camera_history_enable(int on);
void camera_history_view(Cpu *c);
void camera_history_checked(Cpu *c);
void camera_history_accepted(Cpu *c);
int camera_history_continuous(uint64_t frame, const float previous[12], const float current[12]);
