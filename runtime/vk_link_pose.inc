// Link body pose inputs for the shared scene motion field. Only the submit thread changes pose history.
// Each catalog entry identifies one complete index range from the local model archive.
#define POSE_ROWS 256
#define POSE_INDEX_BYTES (512u * 1024u)
#define POSE_BYTES (128u * 1024u)
#define POSE_BANKS 5  // c3 through c7
#define POSE_VIEWS 8
typedef struct {
    uint64_t frame, cam, indices;
    VkImage depth;
    const GfxShader *shader;
    NVNbufferRange vertex[16];
    uint64_t bindings[POSE_BANKS];
    uint32_t size[POSE_BANKS], capacity;
    unsigned char *bytes;
    uint32_t width, height;
    int seen, selected, ambiguous;
} ActorPose;
typedef struct {
    ActorPose view[POSE_VIEWS];
    ActorPose *scene;
} PoseFrame;
typedef struct {
    uint32_t count, width, base, actor;
    unsigned long long hash;
    PoseFrame history[2];
} PoseRange;
static PoseRange pose_ranges[POSE_ROWS];
static int pose_count;
static PlatOnce pose_catalog_once = PLAT_ONCE_INIT;
static uint8_t pose_lengths[POSE_INDEX_BYTES / 8 + 1];
enum { POSE_CHILD = 1, POSE_ADULT = 2 };
static unsigned pose_seen[POSE_ADULT + 1], pose_written[POSE_ADULT + 1], pose_unreadable, pose_missing;

static void pose_load_catalog(void *unused) {
    (void)unused;
    const char *path = getenv("NX2NAVI_LINK_POSE_CATALOG");
    char exe[1024], local[sizeof exe + sizeof "/link-pose.txt"];
    if (!path || !*path) {
        char *sep = plat_exe_path(exe, sizeof exe) ? plat_last_sep(exe) : NULL;
        if (!sep) return;
        *sep = 0;
        snprintf(local, sizeof local, "%s/link-pose.txt", exe);
        path = local;
    }
    FILE *f = fopen(path, "r");
    char line[128];
    if (f && fgets(line, sizeof line, f) && !strcmp(line, "NX2NAVI Link pose catalog 1\n")) {
        while (fgets(line, sizeof line, f)) {
            PoseRange r = {0};
            if (pose_count == POSE_ROWS ||
                sscanf(line, "%u %u %u %llx %u", &r.count, &r.width, &r.base, &r.hash, &r.actor) != 5 ||
                !r.count || (r.width != 2 && r.width != 4) || r.count > POSE_INDEX_BYTES / r.width ||
                r.actor < POSE_CHILD || r.actor > POSE_ADULT || (pose_count && pose_ranges[pose_count-1].count > r.count)) {
                pose_count = 0;
                break;
            }
            pose_ranges[pose_count++] = r;
            uint32_t n = r.count * r.width;
            pose_lengths[n >> 3] |= 1u << (n & 7);
        }
    }
    if (f) fclose(f);
    rt_log("vk_exec: loaded %d animated motion index ranges", pose_count);
}

static int pose_enabled(void) {
    plat_once(&pose_catalog_once, pose_load_catalog, NULL);
    return pose_count != 0;
}

// NVN buffer registration and replay can run on different threads. An entry identifies bytes, so a
// collision only evicts a hint. Validate both pool mappings and writes before using a registered hash.
typedef struct {
    uint64_t addr, mapping, hash;
    uint32_t size, generation;
} PoseBuffer;
static PoseBuffer pose_buffers[2048];
static PlatMutex pose_buffer_lock = PLAT_MUTEX_INIT;

static PoseBuffer *pose_buffer_slot(uint64_t addr, uint32_t size) {
    return &pose_buffers[((addr ^ (uint64_t)size << 32) * 0x9e3779b97f4a7c15ull) >> 53];
}

static uint64_t pose_hash(const uint8_t *data, uint32_t n) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < n; ++i) hash = (hash ^ data[i]) * 0x100000001b3ull;
    return hash;
}

static void pose_remember(PoseBuffer b) {
    plat_mutex_lock(&pose_buffer_lock);
    *pose_buffer_slot(b.addr, b.size) = b;
    plat_mutex_unlock(&pose_buffer_lock);
}

void vk_link_pose_register_buffer(uint64_t addr, uint64_t size) {
    if (!pose_enabled() || !size || size > POSE_INDEX_BYTES ||
        !(pose_lengths[size >> 3] & (1u << (size & 7)))) return;
    PoseBuffer b = {.addr = addr, .size = (uint32_t)size, .mapping = vk_mem_import_identity(addr, size)};
    // Stamp before inspecting the bytes. Writes during or after identification invalidate this observation.
    if (!b.mapping || vk_mem_changed(addr, size, 0, &b.generation) < 0 || !mem_read_hold(addr, size)) return;
    b.hash = pose_hash((const uint8_t *)(uintptr_t)addr, b.size);
    mem_read_release();
    for (int i = 0; i < pose_count; ++i) {
        const PoseRange *r = &pose_ranges[i];
        if (r->count * r->width == size && r->hash == b.hash) {
            pose_remember(b);
            return;
        }
    }
}

static int pose_buffer_hash(uint64_t addr, uint32_t n, uint64_t *hash) {
    int live;
    const uint8_t *bytes = vk_usnap_read_source(addr, n, &live);
    if (!bytes) return 0;
    if (!live) {
        *hash = pose_hash(bytes, n);
        return 1;
    }
    plat_mutex_lock_shared(&pose_buffer_lock);
    PoseBuffer b = *pose_buffer_slot(addr, n);
    plat_mutex_unlock_shared(&pose_buffer_lock);
    uint64_t mapping = vk_mem_import_identity(addr, n);
    if (!mapping) {
        *hash = pose_hash(bytes, n);
        return 1;
    }
    int known = b.addr == addr && b.size == n && b.mapping == mapping;
    uint32_t now;
    int changed = vk_mem_changed(addr, n, known ? b.generation : 0, &now);
    if (known && !changed && now >= b.generation) {
        *hash = b.hash;
    } else {
        *hash = pose_hash(bytes, n);
        if (changed >= 0) {
            pose_remember((PoseBuffer){.addr = addr, .size = n, .mapping = mapping, .generation = now, .hash = *hash});
        } else if (known) {
            // Tracking can disappear with a remap. Drop the older observation during that gap.
            plat_mutex_lock(&pose_buffer_lock);
            pose_buffer_slot(addr, n)->size = 0;
            plat_mutex_unlock(&pose_buffer_lock);
        }
    }
    return 1;
}

static PoseRange *pose_match(const p_draw_idx *d) {
    uint32_t width = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2u : 4u;
    int lo = 0, hi = pose_count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (pose_ranges[mid].count < (uint32_t)d->count) lo = mid + 1;
        else hi = mid;
    }
    uint64_t hash = 0;
    int read_indices = 0;
    for (int i = lo; i < pose_count && pose_ranges[i].count == (uint32_t)d->count; ++i) {
        PoseRange *r = &pose_ranges[i];
        if (r->width != width || r->base != (uint32_t)d->base_vertex) continue;
        if (!read_indices) {
            if (!pose_buffer_hash(d->ibuf, r->count * width, &hash)) { ++pose_unreadable; return NULL; }
            read_indices = 1;
        }
        if (hash == r->hash) { ++pose_seen[r->actor]; return r; }
    }
    return NULL;
}

static ActorPose *pose_previous(PoseRange *r, VkExec *ex, const p_draw_idx *d, uint64_t frame, uint32_t banks,
                                ActorPose **candidate) {
    PoseFrame *now = &r->history[frame & 1];
    ActorPose *prev = r->history[(frame & 1) ^ 1].scene;
    uint64_t cam = ex->ubo[NVN_STAGE_VS][0].addr;
    // The same mesh is drawn into several depth targets. Only the resolve identifies the camera and depth that supply scene motion.
    ActorPose *cur = NULL;
    for (int i = 0; i < POSE_VIEWS; ++i) {
        ActorPose *p = &now->view[i];
        if (!p->seen || p->frame != frame || (p->cam == cam && p->depth == ex->depth.image)) {
            cur = p;
            break;
        }
    }
    *candidate = cur;
    if (!cur) return NULL;
    int repeated = cur->seen && cur->frame == frame;
    if (repeated && cur->ambiguous) return NULL;
    if (!repeated) {
        cur->seen = 1;
        cur->frame = frame;
        cur->selected = cur->ambiguous = 0;
        cur->cam = cam;
        cur->depth = ex->depth.image;
        cur->width = ex->depth.width;
        cur->height = ex->depth.height;
    }

    // Unused bindings retain the previous draw's values. Only the shader's vertex inputs identify this mesh.
    NVNbufferRange vertex[16] = {0};
    uint32_t inputs = gfx_shader_inputs(B.st.vs, NULL);
    for (int i = 0; i < 16; ++i) {
        va_st a;
        memcpy(&a, &B.st.vattr[i], sizeof a);
        if ((inputs >> i & 1) && a.fmt && a.stream < 16) vertex[a.stream] = ex->vbuf[a.stream];
    }
    uint32_t size[POSE_BANKS] = {0}, n = 0;
    uint64_t bindings[POSE_BANKS] = {0};
    const uint8_t *part[POSE_BANKS] = {0};
    for (int i = 0; i < POSE_BANKS; ++i) {
        if (!(banks & (1u << (i + 3)))) continue;
        const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][i];
        VkDescriptorBufferInfo snap;
        if (!u->size || u->size > LIM.ubo_max || u->size > POSE_BYTES - n ||
            !vk_usnap_find(u->addr, u->size, LIM.ubo_align, &snap)) {
            cur->ambiguous = 1;
            ++pose_missing;
            return NULL;
        }
        part[i] = vk_usnap_map() + snap.offset;
        size[i] = (uint32_t)u->size;
        bindings[i] = u->addr;
        n += size[i];
    }
    if (repeated) {
        // The same instance can be submitted again in the same pass. A second instance of this mesh
        // can use another palette, or rewrite a palette at the same address between submissions.
        cur->ambiguous = cur->depth != ex->depth.image || cur->indices != d->ibuf ||
                         memcmp(vertex, cur->vertex, sizeof vertex) || memcmp(size, cur->size, sizeof size) ||
                         memcmp(bindings, cur->bindings, sizeof bindings);
        uint32_t offset = 0;
        for (int i = 0; !cur->ambiguous && i < POSE_BANKS; ++i) {
            if (size[i] && memcmp(cur->bytes + offset, part[i], size[i])) cur->ambiguous = 1;
            offset += size[i];
        }
        return NULL;
    }
    cur->shader = B.st.vs;
    cur->indices = d->ibuf;
    memcpy(cur->vertex, vertex, sizeof vertex);
    memcpy(cur->bindings, bindings, sizeof bindings);
    memcpy(cur->size, size, sizeof size);
    if (n > cur->capacity) {
        uint8_t *larger = realloc(cur->bytes, n);
        if (!larger) { cur->ambiguous = 1; return NULL; }
        cur->bytes = larger;
        cur->capacity = n;
    }
    uint8_t *data = cur->bytes;
    for (int i = 0; i < POSE_BANKS; ++i) {
        if (part[i]) memcpy(data, part[i], cur->size[i]);
        data += cur->size[i];
    }
    return prev && frame && prev->frame == frame - 1 && prev->selected && !prev->ambiguous &&
           prev->width == cur->width && prev->height == cur->height &&
           prev->shader == cur->shader && prev->indices == cur->indices && cur->vertex[0].address &&
           !memcmp(prev->vertex, cur->vertex, sizeof cur->vertex) &&
           !memcmp(prev->size, cur->size, sizeof cur->size) ? prev : NULL;
}

// The first three rows of c6 describe a body part's submitted transform. Compare the rotation
// between poses only when both bases are orthonormal. Translation does not enter the comparison.
static int pose_rotation_vote(const ActorPose *now, const ActorPose *prev) {
    if (!prev || prev->frame + 1 != now->frame || now->size[3] < 48 || prev->size[3] < 48) return -1;
    float m[2][12];
    const ActorPose *p[2] = {now, prev};
    for (int k = 0; k < 2; ++k) {
        memcpy(m[k], p[k]->bytes + p[k]->size[0] + p[k]->size[1] + p[k]->size[2], sizeof m[k]);
        for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
                float dot = 0.0f;
                for (int j = 0; j < 3; ++j) dot += m[k][4 * a + j] * m[k][4 * b + j];
                float wanted = a == b ? 1.0f : 0.0f;
                if (!(dot >= wanted - 0.005f && dot <= wanted + 0.005f)) return -1;
            }
        }
    }
    float trace = 0.0f;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) trace += m[0][4 * i + j] * m[1][4 * i + j];
    // trace(R1 * transpose(R0)) = 1 + 2*cos(angle). In the reversal capture the body turned
    // 33 to 40 degrees in one rendered frame. Its straight-line vectors cut across the silhouette.
    if (trace < 2.879385f) return 2;  // over 20 degrees between rendered poses
    return trace < 2.969616f;         // still turning by over 10 degrees
}

static unsigned pose_fast_turns;

static struct { uint64_t frame; VkImage depth; unsigned hold; } pose_turn;

// Keep generic motion through the reversal. Switching back for a single slower pose in the middle
// makes adjacent frame pairs disagree about the motion of the same silhouette.
static int pose_turn_active(uint64_t frame, VkImage depth, int votes, int fast, int moving) {
    if (pose_turn.frame + 1 != frame || pose_turn.depth != depth || votes < 3) pose_turn.hold = 0;
    pose_turn.frame = frame;
    pose_turn.depth = depth;
    if (votes >= 3 && fast > votes / 2) pose_turn.hold = 6;
    else if (pose_turn.hold && moving > votes / 2) pose_turn.hold = 6;
    else if (pose_turn.hold) --pose_turn.hold;
    return pose_turn.hold != 0;
}

static int pose_select(uint64_t frame, uint64_t camera, VkImage depth) {
    int votes = 0, fast = 0, moving = 0;
    for (int i = 0; i < pose_count; ++i) {
        PoseFrame *h = &pose_ranges[i].history[frame & 1];
        h->scene = NULL;
        for (int j = 0; j < POSE_VIEWS; ++j) {
            ActorPose *p = &h->view[j];
            p->selected = p->seen && p->frame == frame && !p->ambiguous && p->cam == camera && p->depth == depth;
            if (p->selected) {
                h->scene = p;
                int vote = pose_rotation_vote(p, pose_ranges[i].history[(frame & 1) ^ 1].scene);
                if (vote >= 0) {
                    ++votes;
                    fast += vote == 2;
                    moving += vote != 0;
                }
            }
        }
    }
    int fast_turn = pose_turn_active(frame, depth, votes, fast, moving);
    pose_fast_turns += fast_turn;
    if (frame % 120 == 0) {
        rt_log("vk_exec: Link motion matched/written child %u/%u adult %u/%u, unreadable index ranges %u, missing pose blocks %u, rapid-turn frames %u",
               pose_seen[POSE_CHILD], pose_written[POSE_CHILD], pose_seen[POSE_ADULT], pose_written[POSE_ADULT],
               pose_unreadable, pose_missing, pose_fast_turns);
    }
    // Make one decision for the entire body. A single moving limb must not switch the other parts.
    return fast_turn;
}

static void pose_shutdown(void) {
    if (!pose_enabled()) return;
    for (int i = 0; i < pose_count; ++i) {
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < POSE_VIEWS; ++k) free(pose_ranges[i].history[j].view[k].bytes);
        memset(pose_ranges[i].history, 0, sizeof pose_ranges[i].history);
    }
    memset(&pose_turn, 0, sizeof pose_turn);
    pose_fast_turns = 0;
    plat_mutex_lock(&pose_buffer_lock);
    memset(pose_buffers, 0, sizeof pose_buffers);
    plat_mutex_unlock(&pose_buffer_lock);
}
