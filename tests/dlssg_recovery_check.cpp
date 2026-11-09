// Compile with the SDK and Vulkan includes used by tools/build_dlssg.py. Drive the real bridge through its
// callbacks so the restart interval can be checked for failures that valid game frames seldom produce.
#include "../runtime/dlssg/nx2_sl.cpp"
#include <cassert>

struct TestToken final : sl::FrameToken {
    operator uint32_t() const override { return 1; }
};

static TestToken token;
static Nx2SlCamera camera{};
static Nx2SlImage image{};
static bool reject_constants, reject_tags, active, ui_composed;

static void submit(bool with_ui = true) {
    nx2sl_frame(&token, VK_NULL_HANDLE, &image, &image, &image, with_ui ? &image : nullptr, &camera, 1920, 1080, 6);
}

static void expect_restart() {
    for (int i = 0; i < 11; ++i) {
        submit();
        assert(!active);
    }
    submit();
    assert(active && ui_composed);
}

int main() {
    g_available = 1;
    p_token = [](sl::FrameToken *&, const uint32_t *) { return sl::Result::eErrorInvalidParameter; };
    p_constants = [](const sl::Constants &, const sl::FrameToken &, const sl::ViewportHandle &) {
        return reject_constants ? sl::Result::eErrorInvalidParameter : sl::Result::eOk;
    };
    p_tag = [](const sl::FrameToken &, const sl::ViewportHandle &, const sl::ResourceTag *, uint32_t, sl::CommandBuffer *) {
        return reject_tags ? sl::Result::eErrorInvalidParameter : sl::Result::eOk;
    };
    p_dlssg_options = [](const sl::ViewportHandle &, const sl::DLSSGOptions &o) {
        active = o.mode == sl::DLSSGMode::eOn;
        ui_composed = o.enableUserInterfaceRecomposition == sl::Boolean::eTrue;
        if (active) assert(o.numFramesToGenerate == 5);
        return sl::Result::eOk;
    };

    // HUD inputs arrive partway through a load. The first interpolated frame selects the HUD mode.
    for (int i = 0; i < 11; ++i) {
        submit(false);
        assert(!active);
    }
    submit();
    assert(active && ui_composed);
    reject_constants = true;
    submit();
    assert(!active);
    reject_constants = false;
    expect_restart();
    reject_tags = true;
    submit();
    assert(!active);
    reject_tags = false;
    expect_restart();
    nx2sl_frame(nullptr, VK_NULL_HANDLE, &image, &image, &image, &image, &camera, 1920, 1080, 6);
    assert(!active);
    expect_restart();
    return 0;
}
