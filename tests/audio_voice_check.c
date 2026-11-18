/* Synthetic renderer requests and samples. Run with tests/audio_voice_check.py. */
#include <assert.h>
#include "ipc.h"

// Leave the IPC services out of this executable. The fixture drives the same update, prepare and mix functions.
static void audren_manager_handler(IpcSession *, const IpcRequest *, IpcResponse *);
static void audio_device_handler(IpcSession *, const IpcRequest *, IpcResponse *);
static void audio_dmg_handler(IpcSession *, const IpcRequest *, IpcResponse *);
static void audio_sysmix_handler(IpcSession *, const IpcRequest *, IpcResponse *);
static void audio_channels_live(void);
#include "../runtime/audren.c"
#include "../runtime/audio_mixer.c"

unsigned rt_trace;
const char *cfg_get(const char *key) { return !strcmp(key, "NX2NAVI_AUDIO") ? "none" : NULL; }
const char *cfg_value(const char *key) { (void)key; return ""; }
unsigned cfg_version(void) { return 1; }
double cfg_double(const char *key) { (void)key; return 100; }
uint64_t kern_now_ns(void) { return 0; }
void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_fatal(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}
void ipc_backtrace(void) {}
void *ipc_guest(uint64_t addr, uint64_t size) { (void)size; return (void *)(uintptr_t)addr; }
void menu_sound_mix(float *out, uint32_t frames, int channels, int rate) {
    (void)out; (void)frames; (void)channels; (void)rate;
}

typedef struct {
    AudioRenderer renderer;
    ServerVoice voice;
    AudrenVoiceIn in;
    AudrenMixIn mixes[2];
    AudrenChannelIn channels[6];
    AudrenSinkIn sink;
} Fixture;

static void setup(Fixture *f, int frames) {
    memset(f, 0, sizeof *f);
    f->renderer = (AudioRenderer){
        .param = {.sample_rate = 48000, .sample_count = frames, .voice_count = 1, .mix_buffer_count = 4,
                  .sink_count = 1, .submix_count = 1, .splitter_count = 1, .splitter_destination_count = 1},
        .voices = &f->voice, .channels = f->channels, .mixes = f->mixes, .sinks = &f->sink
    };
    f->mixes[0] = (AudrenMixIn){.in_use = 1, .buffer_count = 2, .volume = 1};
    f->channels[0].mix[0] = 1;
    f->sink = (AudrenSinkIn){.type = SINK_DEVICE, .in_use = 1, .input_count = 2, .inputs = {0, 1}};
    f->in = (AudrenVoiceIn){.is_new = 1, .in_use = 1, .play_state = PLAY_STARTED, .sample_format = FMT_PCMF32,
                            .sample_rate = 48000, .pitch = 1, .volume = 1, .channel_count = 1,
                            .splitter_id = SPLITTER_UNUSED};
    for (int i = 0; i < 4; i++) f->in.wave_buffers[i].sent = 1;
    audio_mixer_open(&f->renderer);
}

static void submit(Fixture *f) {
    update_voice(&f->renderer, &f->in);
    f->in.is_new = 0;
    for (int i = 0; i < 4; i++) f->in.wave_buffers[i].sent = 1;
}

static const float *render(Fixture *f) {
    AudioRenderer *r = &f->renderer;
    voices_prepare(r);
    audio_mixer_render(r, (double)r->param.sample_count / r->param.sample_rate);
    advance_voice(r, &f->voice);
    Mixer *m = r->mixer;
    return m->output_scratch;
}

static void loop_check(void) {
    Fixture f;
    setup(&f, 4);
    float data[] = {0, .125f, .25f, .375f, .5f, .625f, .75f, .875f};
    f.in.pitch = .5f;
    f.in.wave_buffers[0] = (AudrenWaveBuffer){
        .address = (uintptr_t)data, .size = sizeof data, .start = 1, .end = 5,
        .loop = 1, .loop_start = 2, .loop_end = 4, .loop_count = 2
    };
    f.in.wave_buffers[1] = (AudrenWaveBuffer){.address = (uintptr_t)data, .size = sizeof data, .start = 7, .end = 8};
    submit(&f);
    const float expected[] = {1, 1.5f, 2, 2.5f, 3, 3.5f, 4, 3, 2, 2.5f, 3, 2.5f, 2, 2.5f, 3, 5};
    for (int block = 0; block < 4; block++) {
        const float *out = render(&f);
        for (int k = 0; k < 4; k++) assert(out[k * 2] == expected[block * 4 + k] / 8);
        assert(f.voice.played == (unsigned)(block + 1) * 2);
        assert(f.voice.consumed == (unsigned)(block == 3));
    }
    assert(f.voice.index == 1 && render(&f)[0] == data[7]);
    audio_mixer_close(&f.renderer);
}

static void adpcm_check(int slot) {
    Fixture f;
    setup(&f, 2);
    int16_t coef[16] = {2048}, context[3] = {0, 1000, 0};
    uint8_t data[8] = {0, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
    f.in.sample_format = FMT_ADPCM;
    f.in.extra_address = (uintptr_t)coef;
    f.in.extra_size = sizeof coef;
    if (slot) f.in.wave_buffers[0].sent = 0;   // consumed empty buffer ahead of slot one
    f.in.wave_buffers[slot] = (AudrenWaveBuffer){
        .address = (uintptr_t)data, .size = sizeof data, .start = 3, .end = 7,
        .context_address = (uintptr_t)context, .context_size = sizeof context,
        .loop = 1, .loop_start = 6, .loop_end = 7, .loop_count = -1
    };
    submit(&f);
    for (int block = 0; block < 3; block++) {
        const float *out = render(&f);
        for (int k = 0; k < 2; k++) {
            int expected = block == 2 ? 1001 : 1001 + 2 * block + k;
            assert(out[2 * k] * 32768 == expected);
        }
        assert(f.voice.consumed == (unsigned)slot);
    }
    f.in.play_state = PLAY_STOPPED;
    submit(&f);
    render(&f);
    f.in.play_state = PLAY_STARTED;
    int next = f.voice.index;
    f.in.wave_buffers[next] = f.in.wave_buffers[slot];
    f.in.wave_buffers[next].sent = 0;
    context[1] = 2000;
    submit(&f);
    assert(render(&f)[0] * 32768 == 2001);
    audio_mixer_close(&f.renderer);
}

static void transition_check(int split) {
    Fixture f;
    setup(&f, 240);
    Mixer *m = f.renderer.mixer;
    float tone = .5f;
    f.in.wave_buffers[0] = (AudrenWaveBuffer){
        .address = (uintptr_t)&tone, .size = sizeof tone, .end = 1, .loop = 1, .loop_count = -1
    };
    f.in.mix_id = 1;
    f.mixes[1] = (AudrenMixIn){.in_use = 1, .volume = 1, .buffer_count = 1,
                               .dest_mix_id = 0, .dest_splitter_id = SPLITTER_UNUSED};
    f.mixes[1].mix[0][1] = 1;
    if (split) {
        f.in.splitter_id = 0;
        m->split[0].ndest = 1;
        m->split[0].ids = calloc(1, sizeof(int));
        m->dest[0] = (SplitDest){.dest = 1, .used = 1, .vol = {1}};
    }
    submit(&f);
    assert(render(&f)[479] == .5f);
    f.in.volume = .5f;
    submit(&f);
    const float *out = render(&f);
    for (int k = 0; k < 240; k++) assert(fabsf(out[2*k+1] - (.5f - .25f*(k+1)/240)) < 1e-6f);

    f.mixes[0].buffer_count = 3;   // the old submix now occupies a different physical buffer
    if (split) {
        f.in.play_state = PLAY_STOPPED;
    } else {
        // Release and replace the voice between render ticks. The new note goes to the left speaker.
        f.in.in_use = 0;
        submit(&f);
        f.in.is_new = f.in.in_use = 1;
        f.in.mix_id = 0;
        f.in.wave_buffers[0].sent = 0;
        tone = -.5f;
    }
    submit(&f);
    out = render(&f);
    for (int k = 0; k < 240; k++) {
        float tail = k < 96 ? .25f * (95-k) / 96 : 0;
        assert(fabsf(out[2*k+1] - tail) < 1e-6f);
        float attack = split ? 0 : -.25f * (k+1) / 240;
        assert(fabsf(out[2*k] - attack) < 1e-6f);
    }
    if (split) assert(f.voice.consumed == 1 && f.voice.played == 0 && f.voice.state == SV_STOPPED);
    assert(render(&f)[1] == 0);
    audio_mixer_close(&f.renderer);
}

static void adpcm_refill_check(void) {
    Fixture f;
    setup(&f, 2);
    int16_t coef[16] = {2048}, context[3] = {0, 100, 0};
    uint8_t data[8] = {0, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
    f.in.sample_format = FMT_ADPCM;
    f.in.extra_address = (uintptr_t)coef;
    f.in.extra_size = sizeof coef;
    f.in.pitch = .5f;
    f.in.wave_buffers[0] = (AudrenWaveBuffer){.address = (uintptr_t)data, .size = sizeof data, .end = 14,
        .context_address = (uintptr_t)context, .context_size = sizeof context};
    submit(&f);
    const float *out = render(&f);
    assert(out[0] * 32768 == 101 && out[2] * 32768 == 101.5f);
    context[1] = 700;
    f.in.wave_buffers[0].sent = 0;
    submit(&f);
    out = render(&f);
    assert(out[0] * 32768 == 701 && out[2] * 32768 == 701.5f);
    // With pitch above one, the last sample of the old buffer may only be needed as predictor history.
    f.in.is_new = 1;
    f.in.pitch = 3;
    f.in.wave_buffers[0].sent = 0;
    f.in.wave_buffers[0].end = 3;
    f.in.wave_buffers[1] = (AudrenWaveBuffer){.address = (uintptr_t)data, .size = sizeof data, .end = 14};
    submit(&f);
    voices_prepare(&f.renderer);
    float decoded[6][MAX_FRAME];
    voice_render(&f.renderer, &f.voice, decoded, 2);
    assert(decoded[0][0] * 32768 == 701 && decoded[0][1] * 32768 == 704);
    audio_mixer_close(&f.renderer);
}

static void route_gain_check(int split) {
    Fixture f;
    setup(&f, 240);
    Mixer *m = f.renderer.mixer;
    float tone = .5f;
    f.in.wave_buffers[0] = (AudrenWaveBuffer){
        .address = (uintptr_t)&tone, .size = sizeof tone, .end = 1, .loop = 1, .loop_count = -1
    };
    float *gains = f.channels[0].mix;
    if (split) {
        f.in.splitter_id = 0;
        m->split[0].ndest = 1;
        m->split[0].ids = calloc(1, sizeof(int));
        m->dest[0] = (SplitDest){.dest = 0, .used = 1, .vol = {1}};
        gains = m->dest[0].vol;
    }
    submit(&f);
    assert(render(&f)[478] == .5f);
    // A held tone changes speaker level through the channel resource, with the voice volume fixed.
    gains[0] = 0;
    gains[1] = 1;
    const float *out = render(&f);
    for (int k = 0; k < 240; k++) {
        float right = .5f * (k + 1) / 240;
        assert(fabsf(out[2*k] - (.5f - right)) < 1e-6f);
        assert(fabsf(out[2*k+1] - right) < 1e-6f);
    }
    out = render(&f);
    assert(out[0] == 0 && out[1] == .5f);
    // Several parameter updates before a render still begin at the last audible level.
    f.in.volume = .75f;
    gains[1] = .75f;
    submit(&f);
    f.in.volume = .5f;
    gains[1] = .5f;
    submit(&f);
    out = render(&f);
    for (int k = 0; k < 240; k++)
        assert(fabsf(out[2*k+1] - (.5f - .375f * (k + 1) / 240)) < 1e-6f);
    gains[1] = 0;
    out = render(&f);
    for (int k = 0; k < 240; k++)
        assert(fabsf(out[2*k+1] - .125f * (1 - (k + 1) / 240.0f)) < 1e-6f);
    assert(render(&f)[1] == 0);
    audio_mixer_close(&f.renderer);
}

int main(void) {
    loop_check();
    adpcm_check(0);
    adpcm_check(1);
    adpcm_refill_check();
    transition_check(0);
    transition_check(1);
    route_gain_check(0);
    route_gain_check(1);
    puts("Audio loop, decoder history and voice transition checks passed.");
    return 0;
}
