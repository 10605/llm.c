// Check that serializing a freshly seeded RNG cannot include uninitialized bytes.
#include <stdio.h>
#include <string.h>
#include "rand.h"

int main(void) {
    mt19937_state a, b;
    memset(&a, 0x55, sizeof(a));
    memset(&b, 0xaa, sizeof(b));
    manual_seed(&a, 137);
    manual_seed(&b, 137);
    if (memcmp(&a, &b, sizeof(a)) || a.seed_ != 137) {
        fprintf(stderr, "Seeded RNG state contains uninitialized metadata\n");
        return 1;
    }
    const unsigned int expected[] = {4053805790u, 2173880614u, 380293709u, 1237255315u};
    for (int i = 0; i < 4; ++i) {
        if (randint32(&a) != expected[i] || randint32(&b) != expected[i]) return 1;
    }
    puts("RNG serialization and original random sequence passed");
    return 0;
}
