#include "hutoken/hash.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
//
// Murmur3_86_128
//-----------------------------------------------------------------------------
static uint64_t MM86128(const void* key, const size_t len) {
#define ROTL32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define FMIX32(h)      \
    h ^= (h) >> 16;    \
    (h) *= 0x85ebca6b; \
    (h) ^= (h) >> 13;  \
    (h) *= 0xc2b2ae35; \
    (h) ^= (h) >> 16;
    const uint8_t* data = (const uint8_t*)key;
    const size_t nblocks = len / 16;
    uint32_t h1 = SEED;
    uint32_t h2 = SEED;
    uint32_t h3 = SEED;
    uint32_t h4 = SEED;
    uint32_t c1 = 0x239b961b;
    uint32_t c2 = 0xab0e9789;
    uint32_t c3 = 0x38b34ae5;
    uint32_t c4 = 0xa1e38b93;
    const uint32_t* blocks =
        (const uint32_t*)(data + (ptrdiff_t)(nblocks * 16));
    for (size_t i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i * 4 + 0];
        uint32_t k2 = blocks[i * 4 + 1];
        uint32_t k3 = blocks[i * 4 + 2];
        uint32_t k4 = blocks[i * 4 + 3];
        k1 *= c1;
        k1 = ROTL32(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        h1 = ROTL32(h1, 19);
        h1 += h2;
        h1 = h1 * 5 + 0x561ccd1b;
        k2 *= c2;
        k2 = ROTL32(k2, 16);
        k2 *= c3;
        h2 ^= k2;
        h2 = ROTL32(h2, 17);
        h2 += h3;
        h2 = h2 * 5 + 0x0bcaa747;
        k3 *= c3;
        k3 = ROTL32(k3, 17);
        k3 *= c4;
        h3 ^= k3;
        h3 = ROTL32(h3, 15);
        h3 += h4;
        h3 = h3 * 5 + 0x96cd1c35;
        k4 *= c4;
        k4 = ROTL32(k4, 18);
        k4 *= c1;
        h4 ^= k4;
        h4 = ROTL32(h4, 13);
        h4 += h1;
        h4 = h4 * 5 + 0x32ac3b17;
    }
    const uint8_t* tail = data + (ptrdiff_t)(nblocks * 16);
    uint32_t k1 = 0;
    uint32_t k2 = 0;
    uint32_t k3 = 0;
    uint32_t k4 = 0;
    switch (len & 15) {
        case 15:
            k4 ^= tail[14] << 16; /* fall through */
        case 14:
            k4 ^= tail[13] << 8; /* fall through */
        case 13:
            k4 ^= tail[12] << 0;
            k4 *= c4;
            k4 = ROTL32(k4, 18);
            k4 *= c1;
            h4 ^= k4;
            /* fall through */
        case 12:
            k3 ^= tail[11] << 24; /* fall through */
        case 11:
            k3 ^= tail[10] << 16; /* fall through */
        case 10:
            k3 ^= tail[9] << 8; /* fall through */
        case 9:
            k3 ^= tail[8] << 0;
            k3 *= c3;
            k3 = ROTL32(k3, 17);
            k3 *= c4;
            h3 ^= k3;
            /* fall through */
        case 8:
            k2 ^= tail[7] << 24; /* fall through */
        case 7:
            k2 ^= tail[6] << 16; /* fall through */
        case 6:
            k2 ^= tail[5] << 8; /* fall through */
        case 5:
            k2 ^= tail[4] << 0;
            k2 *= c2;
            k2 = ROTL32(k2, 16);
            k2 *= c3;
            h2 ^= k2;
            /* fall through */
        case 4:
            k1 ^= tail[3] << 24; /* fall through */
        case 3:
            k1 ^= tail[2] << 16; /* fall through */
        case 2:
            k1 ^= tail[1] << 8; /* fall through */
        case 1:
            k1 ^= tail[0] << 0;
            k1 *= c1;
            k1 = ROTL32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
            /* fall through */
        // This default case is added to make the code's intent explicit
        // to static analyzers. The Murmur3 algorithm correctly does nothing
        // when there is no tail (len & 15 == 0), and this makes that choice
        // clear.
        default:
            break;
    };
    h1 ^= len;
    h2 ^= len;
    h3 ^= len;
    h4 ^= len;
    h1 += h2;
    h1 += h3;
    h1 += h4;
    h2 += h1;
    h3 += h1;
    h4 += h1;
    FMIX32(h1);
    FMIX32(h2);
    FMIX32(h3);
    FMIX32(h4);
    h1 += h2;
    h1 += h3;
    h1 += h4;
    h2 += h1;
    h3 += h1;
    h4 += h1;

    // Discarding the upper 64 bits of the 128-bit hash, because the original
    // author wrote it this way. However, for the correct implementation of
    // the 128-bit hash, both these values would have to be used.
    (void)h3;
    (void)h4;
    return (((uint64_t)h2) << 32) | h1;
}

// hashmap_murmur returns a hash value for `data` using Murmur3_86_128.
uint64_t hashmap_murmur(const void* data, size_t len) {
    return MM86128(data, len);
}
