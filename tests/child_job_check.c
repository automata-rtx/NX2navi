// Each worker answers a pipe request, then stays busy until its owner exits. Holding its process
// handle before ending the owner lets the test observe the exact worker, even if a PID is reused.
#include <stdio.h>
#include <string.h>
#include <windows.h>
#include "platform/platform.h"

static int requested(void) {
    char c;
    DWORD n;
    return ReadFile(GetStdHandle(STD_INPUT_HANDLE), &c, 1, &n, NULL) && n == 1 && c == '?';
}

static int reply_pid(void) {
    DWORD pid = GetCurrentProcessId(), written;
    return WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), &pid, sizeof pid, &written, NULL) && written == sizeof pid;
}

static int get_pid(PlatChild *child, DWORD *pid) {
    return child && plat_child_write(child, "?", 1) && plat_child_read(child, pid, sizeof *pid) == sizeof *pid;
}

static int owner(const char *exe) {
    PlatChild *worker = plat_child_start(exe, "--worker", NULL);
    DWORD pid, n;
    char request;
    if (!requested() || !get_pid(worker, &pid)) return 1;
    if (!WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), &pid, sizeof pid, &n, NULL) || n != sizeof pid) return 1;
    return ReadFile(GetStdHandle(STD_INPUT_HANDLE), &request, 1, &n, NULL) && n == 1 ? 0 : 1;
}

static int check(const char *exe, int normal_exit) {
    PlatChild *parent = plat_child_start(exe, "--owner", NULL);
    DWORD pid;
    HANDLE worker = NULL;
    if (get_pid(parent, &pid)) worker = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, pid);
    int ok = worker && WaitForSingleObject(worker, 0) == WAIT_TIMEOUT;
    if (normal_exit && ok) ok = plat_child_write(parent, ".", 1);
    if (!normal_exit) plat_child_stop(parent);
    if (ok) ok = WaitForSingleObject(worker, 3000) == WAIT_OBJECT_0;
    if (normal_exit) plat_child_stop(parent);
    if (worker) {
        if (!ok) TerminateProcess(worker, 1);
        CloseHandle(worker);
    }
    return ok;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--worker") == 0) {
        if (!requested() || !reply_pid()) return 1;
        Sleep(10000);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--owner") == 0) return owner(argv[0]);
    int normal = check(argv[0], 1), abrupt = check(argv[0], 0);
    printf("Worker exited after owner returned: %d, after owner was terminated: %d\n", normal, abrupt);
    return normal && abrupt ? 0 : 1;
}
