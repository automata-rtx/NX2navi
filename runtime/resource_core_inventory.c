#include "resource_core_inventory.h"
#include "resource_core_shader_writes.h"
#include <limits.h>
#include <stdlib.h>

typedef struct {
    size_t def;
    uint32_t type, storage, set, binding, seen;
} InventoryId;
typedef struct {
    const uint32_t *w;
    uint32_t bound, epoch;
    InventoryId *ids;
    uint32_t *todo;
    RcWriteSiteCovered covered;
    void *context;
} Inventory;

static int typed_result(uint32_t op) {
    return op == 1 || op == 43 || op == 46 || op == 55 || op == 57 ||
           (op >= 59 && op <= 61) || (op >= 65 && op <= 67) || op == 70 ||
           op == 83 || op == 86 || op == 100 ||
           (op >= 120 && op <= 124) || op == 169 || op == 245;
}
static uint32_t pointer_storage(const Inventory *s, uint32_t id) {
    if (id >= s->bound || !s->ids[id].type) return UINT32_MAX;
    uint32_t t = s->ids[id].type;
    if (t >= s->bound || !s->ids[t].def) return UINT32_MAX;
    const uint32_t *p = s->w + s->ids[t].def;
    return (p[0] & 65535) == 32 && p[0] >> 16 == 4 ? p[2] : UINT32_MAX;
}
static int array_count(const Inventory *s, uint32_t variable, uint32_t *count) {
    uint32_t type = s->ids[variable].type;
    if (type >= s->bound || !s->ids[type].def) return 0;
    const uint32_t *p = s->w + s->ids[type].def;
    if ((p[0] & 65535) != 32 || p[0] >> 16 != 4) return 0;
    type = p[3];
    uint64_t n = 1;
    for (uint32_t depth = 0; depth < s->bound; depth++) {
        if (type >= s->bound || !s->ids[type].def) return 0;
        p = s->w + s->ids[type].def;
        uint32_t op = p[0] & 65535;
        if (op == 29) return 0; // Runtime descriptor arrays need a separately proved reachable-element domain.
        if (op != 28) { *count = (uint32_t)n; return 1; }
        if (p[0] >> 16 != 4 || p[3] >= s->bound || !s->ids[p[3]].def) return 0;
        const uint32_t *length = s->w + s->ids[p[3]].def;
        if ((length[0] & 65535) != 43 || length[0] >> 16 != 4 || !length[3]) return 0;
        n *= length[3];
        if (n > UINT32_MAX) return 0;
        type = p[2];
    }
    return 0;
}
static int enqueue(Inventory *s, uint32_t id, uint32_t *n) {
    if (!id || id >= s->bound) return 0;
    if (s->ids[id].seen == s->epoch) return 1;
    s->ids[id].seen = s->epoch;
    s->todo[(*n)++] = id;
    return 1;
}
static int roots_covered(Inventory *s, size_t instruction, uint32_t destination, uint32_t kind) {
    uint32_t n = 0, roots = 0;
    if (++s->epoch == 0) return 0;
    if (!enqueue(s, destination, &n)) return 0;
    for (uint32_t at = 0; at < n; at++) {
        uint32_t id = s->todo[at];
        InventoryId *entry = &s->ids[id];
        if (!entry->def) return 0;
        const uint32_t *p = s->w + entry->def;
        uint32_t op = p[0] & 65535, len = p[0] >> 16;
        if (op == 59) {
            uint32_t storage = pointer_storage(s, id), count;
            if ((kind == RC_INVENTORY_BUFFER && storage != 2 && storage != 12) ||
                (kind == RC_INVENTORY_IMAGE && storage != 0 && storage != 11) ||
                entry->set == UINT32_MAX || entry->binding == UINT32_MAX ||
                !array_count(s, id, &count)) return 0;
            RcWriteBinding b = {kind, entry->set, entry->binding, count};
            if (!s->covered(s->context, instruction, &b)) return 0;
            roots++;
        } else if (op == 245) {
            if (len < 5 || !(len & 1)) return 0;
            for (uint32_t k = 3; k < len; k += 2)
                if (!enqueue(s, p[k], &n)) return 0;
        } else if (op == 169) {
            if (len != 6 || !enqueue(s, p[4], &n) || !enqueue(s, p[5], &n)) return 0;
        } else if ((op >= 65 && op <= 67) || op == 70 || op == 83 ||
                   (op >= 120 && op <= 124)) {
            if (len < 4 || !enqueue(s, p[3], &n)) return 0;
        } else if (kind == RC_INVENTORY_IMAGE && (op == 60 || op == 61 || op == 86 || op == 100)) {
            if (len < 4 || !enqueue(s, p[3], &n)) return 0;
        } else {
            // Pointer values loaded from memory or passed through unanalysed functions do not identify a descriptor.
            return 0;
        }
    }
    return roots != 0;
}

int rc_write_inventory_sites_covered(const uint32_t *w, size_t count,
                                     RcWriteSiteCovered covered, void *context) {
    if (!w || count < 5 || w[0] != 0x07230203 || !w[3] || w[3] > (1u << 20) || !covered) return 0;
    Inventory s = {.w = w, .bound = w[3], .covered = covered, .context = context};
    s.ids = calloc(s.bound, sizeof *s.ids);
    s.todo = malloc((size_t)s.bound * sizeof *s.todo);
    int result = 0;
    if (!s.ids || !s.todo) goto done;
    for (uint32_t k = 0; k < s.bound; k++) s.ids[k].set = s.ids[k].binding = UINT32_MAX;
    for (size_t i = 5; i < count;) {
        uint32_t len = w[i] >> 16, op = w[i] & 65535, id = 0, type = 0;
        if (!len || len > count - i) goto done;
        if (op >= 19 && op <= 39) {
            if (len < 2) goto done;
            id = w[i + 1];
        } else if (typed_result(op)) {
            if (len < 3) goto done;
            type = w[i + 1]; id = w[i + 2];
            if (!type || type >= s.bound) goto done;
        }
        if (id) {
            if (id >= s.bound || s.ids[id].def) goto done;
            s.ids[id].def = i; s.ids[id].type = type;
        }
        if (op == 71 && len >= 3 && (w[i + 2] == 33 || w[i + 2] == 34)) {
            if (len != 4 || !w[i + 1] || w[i + 1] >= s.bound) goto done;
            uint32_t *v = w[i + 2] == 33 ? &s.ids[w[i + 1]].binding : &s.ids[w[i + 1]].set;
            if (*v != UINT32_MAX && *v != w[i + 3]) goto done;
            *v = w[i + 3];
        }
        i += len;
    }
    for (size_t i = 5; i < count; i += w[i] >> 16) {
        uint32_t len = w[i] >> 16, op = w[i] & 65535, at = 0, kind = 0;
        if (op == 99) { at = 1; kind = RC_INVENTORY_IMAGE; }
        else if ((op >= 62 && op <= 64) || op == 228 || op == 319) at = 1;
        else if ((op >= 229 && op <= 242) || op == 318 || op == 5614 || op == 5615 || op == 6035) at = 3;
        else if (op >= 4096 && !rc_shader_extension_no_guest_write(op, len)) goto done;
        if (!at) continue;
        if (len <= at || !w[i + at] || w[i + at] >= s.bound) goto done;
        if (!kind) {
            switch (pointer_storage(&s, w[i + at])) {
            case 2: case 12: kind = RC_INVENTORY_BUFFER; break;
            case 11: kind = RC_INVENTORY_IMAGE; break;
            case 5349: {
                RcWriteBinding b = {.kind = RC_INVENTORY_GPA};
                if (!covered(context, i, &b)) goto done;
                continue;
            }
            case 3: case 4: case 6: case 7: case 5402: continue;
            default: goto done;
            }
        }
        if (!roots_covered(&s, i, w[i + at], kind)) goto done;
    }
    result = 1;
done:
    free(s.ids); free(s.todo);
    return result;
}

typedef struct {
    RcWriteBindingCovered covered;
    void *context;
} InventoryBindingCheck;

static int inventory_binding_covered(void *opaque, size_t instruction, const RcWriteBinding *binding) {
    InventoryBindingCheck *check = opaque;
    (void)instruction;
    return check->covered(check->context, binding);
}

int rc_write_inventory_covered(const uint32_t *words, size_t count,
                               RcWriteBindingCovered covered, void *context) {
    InventoryBindingCheck check = {covered, context};
    return covered && rc_write_inventory_sites_covered(words, count, inventory_binding_covered, &check);
}
